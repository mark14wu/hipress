// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "collective_operations.h"
#include "../logging.h"

namespace horovod {
namespace common {

HorovodOp::HorovodOp(HorovodGlobalState* global_state) : global_state_(global_state) {}

int64_t HorovodOp::NumElements(std::vector<TensorTableEntry>& entries) {
  int64_t num_elements = 0;
  for (auto& e : entries) {
    num_elements += e.num_elem > 0 ? e.num_elem : e.tensor->shape().num_elements();
  }
  return num_elements;
}

// Allreduce
AllreduceOp::AllreduceOp(HorovodGlobalState* global_state) : HorovodOp(global_state) {}

void AllreduceOp::MemcpyInFusionBuffer(const std::vector<TensorTableEntry>& entries, const void*& fused_input_data,
                                       void*& buffer_data, size_t& buffer_len) {
  // Access the fusion buffer.
  auto& first_entry = entries[0];
  auto& buffer = global_state_->fusion_buffer.GetBuffer(
      first_entry.device, first_entry.context->framework());
  buffer_data = const_cast<void*>(buffer->AccessData(first_entry.context));

  int64_t offset = 0;
  for (auto& e : entries) {
    void* buffer_data_at_offset = (uint8_t*) buffer_data + offset;
    MemcpyEntryInFusionBuffer(entries, e, buffer_data_at_offset);
    offset += e.tensor->size();
  }

  buffer_len = (size_t) offset;

  // Set the input data to originate from the buffer.
  fused_input_data = buffer_data;
}

void AllreduceOp::MemcpyOutFusionBuffer(const void* buffer_data, std::vector<TensorTableEntry>& entries) {
  int64_t offset = 0;
  for (auto& e : entries) {
    void* buffer_data_at_offset = (uint8_t*) buffer_data + offset;
    MemcpyEntryOutFusionBuffer(entries, buffer_data_at_offset, e);
    offset += e.tensor->size();
  }
}

// Allgather
AllgatherOp::AllgatherOp(HorovodGlobalState* global_state) : HorovodOp(global_state) {}

Status AllgatherOp::AllocateOutput(std::vector<TensorTableEntry>& entries, const Response& response,
                                   int64_t**& entry_component_sizes, int*& recvcounts) {
  for (size_t ec = 0; ec < entries.size(); ++ec) {
    auto& e = entries[ec];
    // Every tensor participating in Allgather operation may have different
    // first dimension size, but the rest of dimensions are same for all
    // tensors.  Here we get shape of tensor sliced by first dimension.
    TensorShape single_slice_shape;
    for (int i = 1; i < e.tensor->shape().dims(); ++i) {
      single_slice_shape.AddDim(e.tensor->shape().dim_size(i));
    }

    // Copy tensor sizes from the MPI response into a vector of int64_t
    // and compute total size.  This is size of first dimension.
    int64_t total_entry_dimension_size = 0;
    const auto& tensor_sizes = response.tensor_sizes();
    for (int rc = 0; rc < global_state_->size; ++rc) {
      auto component_size = tensor_sizes[ec * global_state_->size + rc];
      total_entry_dimension_size += component_size;
      recvcounts[rc] += component_size * single_slice_shape.num_elements();
      entry_component_sizes[ec][rc] =
          component_size * single_slice_shape.num_elements();
    }

    // Allgather output will have shape of:
    // (sum of first dimension of every tensor) x (tensor slice shape).
    TensorShape output_shape;
    output_shape.AddDim((int64_t) total_entry_dimension_size);
    output_shape.AppendShape(single_slice_shape);

    Status status = e.context->AllocateOutput(output_shape, &e.output);
    if (!status.ok()) {
      return status;
    }
  }

  return Status::OK();
}

void AllgatherOp::SetDisplacements(const int* recvcounts, int*& displcmnts) {
  for (int rc = 0; rc < global_state_->size; ++rc) {
    if (rc == 0) {
      displcmnts[rc] = 0;
    } else {
      displcmnts[rc] = displcmnts[rc - 1] + recvcounts[rc - 1];
    }
  }
}

void AllgatherOp::SetEntryComponentOffsets(const std::vector<TensorTableEntry>& entries,
                                           const int64_t* const* entry_component_sizes,
                                           const int* recvcounts,
                                           int64_t**& entry_component_offsets) {
  unsigned int rank_displacement = 0;
  for (int rc = 0; rc < global_state_->size; ++rc) {
    for (size_t ec = 0; ec < entries.size(); ++ec) {
      if (ec == 0) {
        entry_component_offsets[ec][rc] = rank_displacement;
      } else {
        entry_component_offsets[ec][rc] =
            entry_component_offsets[ec - 1][rc] +
            entry_component_sizes[ec - 1][rc];
      }
    }
    rank_displacement += recvcounts[rc];
  }
}

void AllgatherOp::MemcpyInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                       const int* displcmnts, int element_size, void*& buffer_data) {
  // Access the fusion buffer.
  auto& first_entry = entries[0];
  auto& buffer = global_state_->fusion_buffer.GetBuffer(
      first_entry.device, first_entry.context->framework());
  buffer_data = const_cast<void*>(buffer->AccessData(first_entry.context));

  int64_t offset = displcmnts[global_state_->rank] * element_size;
  for (auto& e : entries) {
    void* buffer_data_at_offset = (uint8_t*) buffer_data + offset;
    std::memcpy(buffer_data_at_offset, e.tensor->data(), (size_t) e.tensor->size());
    offset += e.tensor->size();
  }
}

void AllgatherOp::MemcpyOutFusionBuffer(const int64_t* const* entry_component_offsets,
                                        const int64_t* const* entry_component_sizes, const void* buffer_data,
                                        int element_size, std::vector<TensorTableEntry>& entries) {
  // Copy memory out of the fusion buffer.
  for (size_t ec = 0; ec < entries.size(); ++ec) {
    auto& e = entries[ec];
    int64_t copy_offset = 0;
    for (int rc = 0; rc < global_state_->size; ++rc) {
      int64_t entry_offset = entry_component_offsets[ec][rc] * element_size;
      int64_t entry_size = entry_component_sizes[ec][rc] * element_size;
      std::memcpy((void*) ((uint8_t*) e.output->data() + copy_offset),
                  (void*) ((uint8_t*) buffer_data + entry_offset),
                  (size_t) entry_size);
      copy_offset += entry_size;
    }
  }
}

BroadcastOp::BroadcastOp(HorovodGlobalState* global_state) : HorovodOp(global_state) {}

void BroadcastOp::MemcpyInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                       void*& buffer_data, int element_size) {
  // Access the fusion buffer.
  auto& first_entry = entries[0];
  auto& buffer = global_state_->fusion_buffer.GetBuffer(
      first_entry.device, first_entry.context->framework());
  buffer_data = const_cast<void*>(buffer->AccessData(first_entry.context));

  int64_t offset = 0;
  int64_t byte_size_of_entry;
  for (auto& e : entries) {
    void* buffer_data_at_offset = (uint8_t*) buffer_data + offset;
    byte_size_of_entry = e.num_elem > 0 ? e.num_elem*element_size : e.tensor->size();
    std::memcpy(buffer_data_at_offset, e.tensor->data(), (size_t) byte_size_of_entry);
    offset += byte_size_of_entry;
  }
}

void BroadcastOp::MemcpyOutFusionBuffer(const void* buffer_data,
                                        std::vector<TensorTableEntry>& entries, int element_size) {
  int64_t offset = 0;
  int64_t byte_size_of_entry;
  for (auto& e : entries) {
    void* buffer_data_at_offset = (uint8_t*) buffer_data + offset;
    byte_size_of_entry = e.num_elem > 0 ? e.num_elem*element_size : e.tensor->size();
    std::memcpy((void*) ((uint8_t*) e.output->data()), buffer_data_at_offset, (size_t) byte_size_of_entry);
    offset += byte_size_of_entry;
  }
}

GatherBcastOp::GatherBcastOp(HorovodGlobalState* global_state) : HorovodOp(global_state) {}

GatherOp::GatherOp(HorovodGlobalState* global_state) : HorovodOp(global_state) {}

void GatherOp::SetRecvcountsAndDisplcmnts(std::vector<TensorTableEntry>& entries, const Response& response,
                                int64_t**& entry_component_sizes, int*& recvcounts, int*& displcmnts) {
  for (size_t ec = 0; ec < entries.size(); ++ec) {
    auto& e = entries[ec];
    // Every tensor participating in gather operation may have different
    // first dimension size, but the rest of dimensions are same for all
    // tensors.  Here we get shape of tensor sliced by first dimension.
    TensorShape single_slice_shape;
    for (int i = 1; i < e.tensor->shape().dims(); ++i) {
      single_slice_shape.AddDim(e.tensor->shape().dim_size(i));
    }

    // Copy tensor sizes from the MPI response into a vector of int64_t
    // and compute total size.  This is size of first dimension.
    const auto& tensor_sizes = response.tensor_sizes();
    for (int rc = 0; rc < global_state_->size; ++rc) {
      auto component_size = tensor_sizes[ec * global_state_->size + rc];
      if (single_slice_shape.num_elements() == 0) {
        recvcounts[rc] += component_size;
        entry_component_sizes[ec][rc] = component_size;
      } else {
        recvcounts[rc] += component_size * single_slice_shape.num_elements();
        entry_component_sizes[ec][rc] =
            component_size * single_slice_shape.num_elements();
      }
    }
  }
  for (int rc = 0; rc < global_state_->size; ++rc) {
    if (rc == 0) {
      displcmnts[rc] = 0;
    } else {
      displcmnts[rc] = displcmnts[rc - 1] + recvcounts[rc - 1];
    }
  }
}

void GatherOp::SetEntryComponentOffsets(const std::vector<TensorTableEntry>& entries,
                                        const int64_t* const* entry_component_sizes,
                                        const int* recvcounts,
                                        int64_t**& entry_component_offsets) {
  unsigned int rank_displacement = 0;
  for (int rc = 0; rc < global_state_->size; ++rc) {
    for (size_t ec = 0; ec < entries.size(); ++ec) {
      if (ec == 0) {
        entry_component_offsets[ec][rc] = rank_displacement;
      } else {
        entry_component_offsets[ec][rc] =
            entry_component_offsets[ec - 1][rc] +
            entry_component_sizes[ec - 1][rc];
      }
    }
    rank_displacement += recvcounts[rc];
  }
}

void GatherOp::MemcpyInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                    void*& buffer_data, int element_size, int64_t offset) {
  // Access the fusion buffer.
  auto& first_entry = entries[0];
  auto& buffer = global_state_->fusion_buffer.GetBuffer(first_entry.device, first_entry.context->framework());
  buffer_data = const_cast<void*>(buffer->AccessData(first_entry.context));

  int64_t byte_size_of_entry;
  for (auto& e : entries) {
    void* buffer_data_at_offset = (uint8_t*) buffer_data + offset;
    byte_size_of_entry = e.num_elem > 0 ? e.num_elem*element_size : e.tensor->size();
    std::memcpy(buffer_data_at_offset, e.tensor->data(), (size_t) byte_size_of_entry);
    offset += byte_size_of_entry;
  }
}

void GatherOp::MemcpyOutFusionBuffer(const int64_t* const* entry_component_offsets,
                                     const int64_t* const* entry_component_sizes, const void* buffer_data,
                                     int element_size, std::vector<TensorTableEntry>& entries) {
  // Copy memory out of the fusion buffer.
  for (size_t ec = 0; ec < entries.size(); ++ec) {
    auto& e = entries[ec];
    int64_t copy_offset = 0;
    for (int rc = 0; rc < global_state_->size; ++rc) {
      int64_t entry_offset = entry_component_offsets[ec][rc] * element_size;
      int64_t entry_size = entry_component_sizes[ec][rc] * element_size;
      if (rc != e.root_rank) {
        std::memcpy((void*) ((uint8_t*) e.output->data() + copy_offset),
                    (void*) ((uint8_t*) buffer_data + entry_offset),
                    (size_t) entry_size);
      }
      copy_offset += entry_size;
    }
  }
}

AlltoallOp::AlltoallOp(HorovodGlobalState* global_state) : HorovodOp(global_state) {}

void AlltoallOp::SetRecvcountsAndDisplcmnts(std::vector<TensorTableEntry>& entries,
                                            const Response& response,
                                            const int roots_count,
                                            int64_t**& entry_component_sizes,
                                            int*& recvcounts,
                                            int*& displcmnts) {
  for (size_t ec = 0; ec < response.entries_counts()[roots_count]; ++ec) {
    auto start_offset = response.entries_offsets()[roots_count];
    auto& e = entries[start_offset+ec];
    // Every tensor participating in gather operation may have different
    // first dimension size, but the rest of dimensions are same for all
    // tensors.  Here we get shape of tensor sliced by first dimension.
    TensorShape single_slice_shape;
    for (int i = 1; i < e.tensor->shape().dims(); ++i) {
      single_slice_shape.AddDim(e.tensor->shape().dim_size(i));
    }

    // Copy tensor sizes from the MPI response into a vector of int64_t
    // and compute total size.  This is size of first dimension.
    const auto& tensor_sizes = response.tensor_sizes();
    for (int rc = 0; rc < global_state_->size; ++rc) {
      auto component_size = tensor_sizes[start_offset * global_state_->size + ec * global_state_->size + rc];
      if (single_slice_shape.num_elements() == 0) {
        recvcounts[rc] += component_size;
        entry_component_sizes[ec][rc] = component_size;
      } else {
        recvcounts[rc] += component_size * single_slice_shape.num_elements();
        entry_component_sizes[ec][rc] =
            component_size * single_slice_shape.num_elements();
      }
    }
  }
  for (int rc = 0; rc < global_state_->size; ++rc) {
    if (rc == 0) {
      displcmnts[rc] = 0;
    } else {
      displcmnts[rc] = displcmnts[rc - 1] + recvcounts[rc - 1];
    }
  }
}

void AlltoallOp::SetSendcounts(std::vector<TensorTableEntry>& entries,
                                            const Response& response,
                                            const int root_rank,
                                            const int roots_count,
                                            int*& sendcounts) {
  // set the sendcounts[root_rank]
  int num_elements = 0;
  auto start_offset = response.entries_offsets()[roots_count];
  for (size_t ec = 0; ec < response.entries_counts()[roots_count]; ec++) {
    auto& e = entries[start_offset+ec];
    num_elements += e.num_elem > 0 ? e.num_elem : e.tensor->shape().num_elements();
  }
  sendcounts[root_rank] = num_elements;
}

void AlltoallOp::SetEntryComponentOffsets(const std::vector<TensorTableEntry>& entries,
                                          const int32_t entries_count,
                                          const int64_t* const* entry_component_sizes,
                                          const int* recvcounts,
                                          int64_t**& entry_component_offsets) {
  unsigned int rank_displacement = 0;
  for (int rc = 0; rc < global_state_->size; ++rc) {
    for (size_t ec = 0; ec < entries_count; ++ec) {
      if (ec == 0) {
        entry_component_offsets[ec][rc] = rank_displacement;
      } else {
        entry_component_offsets[ec][rc] =
            entry_component_offsets[ec - 1][rc] +
            entry_component_sizes[ec - 1][rc];
      }
    }
    rank_displacement += recvcounts[rc];
  }
}

void AlltoallOp::MemcpyInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                      const Response& response,
                                      const int element_size,
                                      int*& send_counts,
                                      int*& send_displcmnts,
                                      void*& buffer_data) {
  // Access the fusion buffer.
  auto& first_entry = entries[0];
  auto& buffer = global_state_->fusion_buffer.GetSendBuffer(first_entry.device, first_entry.context->framework());
  buffer_data = const_cast<void*>(buffer->AccessData(first_entry.context));

  int64_t byte_size_of_entry;
  int64_t offset = 0;
  for (int i = 0; i < global_state_->size; i++) {
    int64_t send_count = 0;
    if (send_counts[i] > 0) { // this node will send data
      // copy the entries to send fusion buffer
      offset = send_displcmnts[i];
      auto& root_ranks = response.root_ranks();
      auto itr = std::find(root_ranks.begin(), root_ranks.end(), i);
      assert(itr != root_ranks.end());
      int rank_idx = std::distance(root_ranks.begin(), itr);
      for (size_t ec = 0; ec < response.entries_counts()[rank_idx]; ec++) {
        auto& e = entries[response.entries_offsets()[rank_idx]+ec];
        void* buffer_data_at_offset = (uint8_t*) buffer_data + offset;
        byte_size_of_entry = e.num_elem > 0 ? e.num_elem*element_size : e.tensor->size();
        std::memcpy(buffer_data_at_offset, e.tensor->data(), (size_t) byte_size_of_entry);
        offset += byte_size_of_entry;
        send_count += e.num_elem > 0 ? e.num_elem : e.tensor->shape().num_elements();
      }
    }
    assert(send_count == send_counts[i]);
  }
}

void AlltoallOp::MemcpyOutFusionBuffer(const int64_t* const* entry_component_offsets,
                                       const int64_t* const* entry_component_sizes,
                                       const void* buffer_data,
                                       const Response& response,
                                       int element_size,
                                       std::vector<TensorTableEntry>& entries) {
  auto& root_ranks = response.root_ranks();
  auto itr = std::find(root_ranks.begin(), root_ranks.end(), global_state_->rank);
  assert(itr != root_ranks.end());
  int rank_idx = std::distance(root_ranks.begin(), itr);
  // copy from gathered fusion buffer to entries
  for (size_t ec = 0; ec < response.entries_counts()[rank_idx]; ec++) {
    auto& e = entries[response.entries_offsets()[rank_idx]+ec];
    int64_t copy_offset = 0;
    for (int rc = 0; rc < global_state_->size; rc++) {
      int64_t entry_offset = entry_component_offsets[ec][rc] * element_size;
      int64_t entry_size = entry_component_sizes[ec][rc] * element_size;
      if (rc != e.root_rank) {
        std::memcpy((void*) ((uint8_t*) e.output->data() + copy_offset),
                    (void*) ((uint8_t*) buffer_data + entry_offset),
                    (size_t) entry_size);
      }
      copy_offset += entry_size;
    }
  }
}

ErrorOp::ErrorOp(HorovodGlobalState* global_state) : HorovodOp(global_state) {}

Status ErrorOp::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  return Status::PreconditionError(response.error_message());
}

} // namespace common
} // namespace horovod
