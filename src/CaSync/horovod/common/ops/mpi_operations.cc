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

#include "mpi_operations.h"
#include "../logging.h"

namespace horovod {
namespace common {

MPIAllreduce::MPIAllreduce(MPIContext* mpi_context, HorovodGlobalState* global_state)
    : AllreduceOp(global_state), mpi_context_(mpi_context) {}

Status MPIAllreduce::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  auto& first_entry = entries[0];

  void* buffer_data;
  size_t buffer_len;
  int64_t num_elements = NumElements(entries);

  // Copy memory into the fusion buffer.
  auto& timeline = global_state_->timeline;
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    const void* fused_input_data;
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);
    timeline.ActivityEndAll(entries);
  } else {
    buffer_data = (void*) first_entry.output->data();
    buffer_len = (size_t) first_entry.output->size();
  }

  // Do allreduce.
  timeline.ActivityStartAll(entries, MPI_ALLREDUCE);
  const void* sendbuf = entries.size() > 1 || first_entry.tensor->data() == first_entry.output->data()
                        ? MPI_IN_PLACE : first_entry.tensor->data();
  int op = MPI_Allreduce(sendbuf, buffer_data,
                         (int) num_elements,
                         mpi_context_->GetMPIDataType(first_entry.tensor),
                         mpi_context_->GetMPISumOp(first_entry.tensor->dtype()),
                         mpi_context_->GetMPICommunicator(Communicator::GLOBAL));
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Allreduce failed, see MPI output for details.");
  }
  timeline.ActivityEndAll(entries);

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(buffer_data, entries);
    timeline.ActivityEndAll(entries);
  }

  return Status::OK();
}

bool MPIAllreduce::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

void MPIAllreduce::MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                             const TensorTableEntry& e, void* buffer_data_at_offset) {
  std::memcpy(buffer_data_at_offset, e.tensor->data(),
              (size_t) e.tensor->size());
}

void MPIAllreduce::MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                              const void* buffer_data_at_offset, TensorTableEntry& e) {
  std::memcpy((void*) e.output->data(), buffer_data_at_offset,
              (size_t) e.tensor->size());
}

MPIAllgather::MPIAllgather(MPIContext* mpi_context, HorovodGlobalState* global_state)
    : AllgatherOp(global_state), mpi_context_(mpi_context) {}

bool MPIAllgather::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  // return true;
  return false; // there are ncclallgather, disable mpi allgather
}

Status MPIAllgather::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  auto& timeline = global_state_->timeline;

  // Sizes of subcomponents of each entry from all ranks
  auto** entry_component_sizes = new int64_t* [entries.size()];

  // Offset of each subcomponent of every entry in the final buffer after
  // allgatherv
  auto** entry_component_offsets = new int64_t* [entries.size()];

  auto* recvcounts = new int[global_state_->size]();
  auto* displcmnts = new int[global_state_->size]();

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    entry_component_sizes[ec] = new int64_t[global_state_->size]();
    entry_component_offsets[ec] = new int64_t[global_state_->size]();
  }

  auto& first_entry = entries[0];

  timeline.ActivityStartAll(entries, ALLOCATE_OUTPUT);
  Status status = AllocateOutput(entries, response, entry_component_sizes, recvcounts);
  if (!status.ok()) {
    return status;
  }
  timeline.ActivityEndAll(entries);

  SetDisplacements(recvcounts, displcmnts);
  SetEntryComponentOffsets(entries, entry_component_sizes, recvcounts, entry_component_offsets);

  int element_size = mpi_context_->GetMPITypeSize(first_entry.tensor->dtype());

  const void* sendbuf = nullptr;
  void* buffer_data;
  int64_t total_num_elements = NumElements(entries);

  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    MemcpyInFusionBuffer(entries, displcmnts, element_size, buffer_data);
    timeline.ActivityEndAll(entries);
  } else {
    sendbuf = first_entry.tensor->data();
    buffer_data = (void*) first_entry.output->data();
  }

  global_state_->timeline.ActivityStartAll(entries, MPI_ALLGATHER);
  auto dtype = mpi_context_->GetMPIDataType(first_entry.tensor->dtype());
  int op = MPI_Allgatherv(sendbuf != nullptr ? sendbuf : MPI_IN_PLACE,
                          (int) total_num_elements,
                          dtype,
                          buffer_data,
                          recvcounts,
                          displcmnts,
                          dtype,
                          mpi_context_->GetMPICommunicator(Communicator::GLOBAL));
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Allgatherv failed, see MPI output for details.");
  }
  global_state_->timeline.ActivityEndAll(entries);

  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(entry_component_offsets, entry_component_sizes,
                          buffer_data, element_size, entries);
    timeline.ActivityEndAll(entries);
  }

  delete[] recvcounts;
  delete[] displcmnts;

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    delete[] entry_component_sizes[ec];
    delete[] entry_component_offsets[ec];
  }
  delete[] entry_component_sizes;
  delete[] entry_component_offsets;

  return Status::OK();
}

MPIHierarchicalAllgather::MPIHierarchicalAllgather(MPIContext* mpi_context,
                                                   HorovodGlobalState* global_state)
    : MPIAllgather(mpi_context, global_state) {}

Status MPIHierarchicalAllgather::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  auto& timeline = global_state_->timeline;

  // Sizes of subcomponents of each entry from all ranks
  auto** entry_component_sizes = new int64_t* [entries.size()];

  // Offset of each subcomponent of every entry in the final buffer after
  // allgatherv
  auto** entry_component_offsets = new int64_t* [entries.size()];

  auto* recvcounts = new int[global_state_->size]();
  auto* displcmnts = new int[global_state_->size]();

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    entry_component_sizes[ec] = new int64_t[global_state_->size]();
    entry_component_offsets[ec] = new int64_t[global_state_->size]();
  }

  auto& first_entry = entries[0];

  timeline.ActivityStartAll(entries, ALLOCATE_OUTPUT);
  Status status = AllocateOutput(entries, response, entry_component_sizes, recvcounts);
  if (!status.ok()) {
    return status;
  }
  timeline.ActivityEndAll(entries);

  SetDisplacements(recvcounts, displcmnts);
  SetEntryComponentOffsets(entries, entry_component_sizes, recvcounts, entry_component_offsets);

  int element_size = mpi_context_->GetMPITypeSize(first_entry.tensor->dtype());

  int64_t total_size = displcmnts[global_state_->size - 1] +
                       recvcounts[global_state_->size - 1];

  // If shared buffer is not initialized or is not large enough, reallocate
  int64_t total_size_in_bytes = total_size * element_size;
  if (global_state_->shared_buffer == nullptr || global_state_->shared_buffer_size < total_size_in_bytes) {
    if (global_state_->shared_buffer != nullptr) {
      MPI_Win_fence(0, mpi_context_->window);
      MPI_Win_free(&mpi_context_->window);
      global_state_->shared_buffer = nullptr;
    }

    // Allocate shared memory, give each rank their respective pointer
    timeline.ActivityStartAll(entries, ALLOCATE_SHARED_BUFFER);
    int64_t window_size = global_state_->local_rank == 0 ? total_size_in_bytes : 0;
    MPI_Win_allocate_shared(window_size,
                            element_size,
                            MPI_INFO_NULL,
                            mpi_context_->GetMPICommunicator(Communicator::LOCAL),
                            &global_state_->shared_buffer,
                            &mpi_context_->window);
    if (global_state_->local_rank != 0) {
      int disp_unit;
      MPI_Aint winsize;
      MPI_Win_shared_query(mpi_context_->window,
                           0,
                           &winsize,
                           &disp_unit,
                           &global_state_->shared_buffer);
    }
    global_state_->shared_buffer_size = total_size_in_bytes;
    timeline.ActivityEndAll(entries);
  }

  // Compute cross-node allgather displacements and recvcounts for
  // homogeneous/parallelized case
  auto* cross_recvcounts = new int[global_state_->cross_size]();
  auto* cross_displcmnts = new int[global_state_->cross_size]();

  if (global_state_->is_homogeneous) {
    for (int i = 0; i < global_state_->cross_size; ++i) {
      cross_recvcounts[i] = recvcounts[global_state_->local_size * i +
                                       global_state_->local_rank];
      cross_displcmnts[i] = displcmnts[global_state_->local_size * i +
                                       global_state_->local_rank];
    }
  } else if (global_state_->local_rank == 0) {
    // In this case local rank 0 will allgather with all local data
    int offset = 0;
    for (int i = 0; i < global_state_->cross_size; ++i) {
      for (int j = offset; j < offset + global_state_->local_sizes[i];
           ++j) {
        cross_recvcounts[i] += recvcounts[j];
      }
      cross_displcmnts[i] = displcmnts[offset];
      offset += global_state_->local_sizes[i];
    }
  }

  timeline.ActivityStartAll(entries, MEMCPY_IN_SHARED_BUFFER);
  for (size_t ec = 0; ec < entries.size(); ++ec) {
    auto& e = entries[ec];
    void* shared_buffer_at_offset =
        (uint8_t*) global_state_->shared_buffer +
        entry_component_offsets[ec][global_state_->rank] * element_size;

    // CPU copy to shared buffer
    memcpy(shared_buffer_at_offset, e.tensor->data(),
           (size_t) (entry_component_sizes[ec][global_state_->rank] *
                     element_size));
  }
  Barrier();
  timeline.ActivityEndAll(entries);

  // Perform the cross-node allgather. If the cluster is homogeneous all
  // local ranks participate, otherwise local rank 0 handles all data
  global_state_->timeline.ActivityStartAll(entries, MPI_CROSS_ALLGATHER);
  if (global_state_->is_homogeneous || global_state_->local_rank == 0) {
    int op = MPI_Allgatherv(MPI_IN_PLACE,
                            0,
                            MPI_DATATYPE_NULL,
                            global_state_->shared_buffer,
                            cross_recvcounts,
                            cross_displcmnts,
                            mpi_context_->GetMPIDataType(first_entry.tensor->dtype()),
                            mpi_context_->GetMPICommunicator(Communicator::CROSS));
    if (op != MPI_SUCCESS) {
      throw std::logic_error("MPI_Allgatherv failed, see MPI output for details.");
    }
  }
  Barrier();
  global_state_->timeline.ActivityEndAll(entries);

  // Copy memory out of the fusion buffer.
  timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
  MemcpyOutFusionBuffer(entry_component_offsets, entry_component_sizes,
                        global_state_->shared_buffer, element_size, entries);
  Barrier();
  timeline.ActivityEndAll(entries);

  // Free the buffers
  delete[] cross_displcmnts;
  delete[] cross_recvcounts;

  return Status::OK();
}

bool MPIHierarchicalAllgather::Enabled(const ParameterManager& param_manager,
                                       const std::vector<TensorTableEntry>& entries,
                                       const Response& response) const {
  // return param_manager.HierarchicalAllgather();
  return false; //there are nccl allgather, disable mpi allgather
}

void MPIHierarchicalAllgather::Barrier() {
  int op = MPI_Barrier(mpi_context_->GetMPICommunicator(Communicator::GLOBAL));
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Barrier failed, see MPI output for details.");
  }
}

MPIBroadcast::MPIBroadcast(MPIContext* mpi_context, HorovodGlobalState* global_state)
    : BroadcastOp(global_state), mpi_context_(mpi_context) {}

Status MPIBroadcast::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  LOG(DEBUG, global_state_->rank) << "bcast entries size: " << entries.size();
  
  auto e = entries[0];
  for (auto& name : response.tensor_names()) {
    global_state_->timestamp.Start(name, "BBCAST", std::to_string(e.batchid));
  }

  // On root rank, MPI_Bcast sends data, on other ranks it receives data.
  void* data_ptr;
  int64_t num_elements = NumElements(entries);

  int element_size = mpi_context_->GetMPITypeSize(e.tensor->dtype());
  LOG(DEBUG, global_state_->rank) << "bcast response names: " << response.tensor_names_string() << " numelem: " << e.tensor->shape().num_elements() << "elem_size: " << element_size << " tensor name: " << e.tensor_name;
  if (global_state_->rank == e.root_rank) {
    if (entries.size() > 1) {
      MemcpyInFusionBuffer(entries, data_ptr, element_size);
    } else {
      data_ptr = (void*) e.tensor->data();
    }
  } else {
    if (entries.size() > 1) {
      // data_ptr = fusion_buffer
      auto& buffer = global_state_->fusion_buffer.GetBuffer(e.device, e.context->framework());
      data_ptr = const_cast<void*>(buffer->AccessData(e.context));
    } else {
      data_ptr = (void*) e.output->data();
    }
  }

  // {
  //   uint8_t* p = reinterpret_cast<uint8_t*>(data_ptr);
  //   LOG(WARNING) << "\033[33mtensor.name=" << e.tensor_name << "\t[HEADER: " << p[0]+0 << "|" << p[1]+0 \
  //     << "|" << p[2]+0 << "|" << p[3]+0 << "]\033[0m" << std::endl;
  //   // printf("[HEADER]: [%d|%d|%d|%d]\n", p[0]+0, p[1]+0, p[2]+0, p[3]+0);
  // }
  global_state_->timeline.ActivityStartAll(entries, MPI_BCAST);
  int op = MPI_Bcast(data_ptr,
                     (int) num_elements,
                     mpi_context_->GetMPIDataType(e.tensor->dtype()),
                     e.root_rank,
                     mpi_context_->GetMPICommunicator(Communicator::GLOBAL));
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Broadcast failed, see MPI output for details.");
  }

  // deadlock that we've been seeing without it.
  int barrier_op = MPI_Barrier(mpi_context_->GetMPICommunicator(Communicator::GLOBAL));
  if (barrier_op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Barrier failed, see MPI output for details.");
  }

  // {
  //   uint8_t* p = reinterpret_cast<uint8_t*>(data_ptr);
  //   LOG(WARNING) << "\033[33mtensor.name=" << e.tensor_name << "\t[HEADER: " << p[0]+0 << "|" << p[1]+0 \
  //     << "|" << p[2]+0 << "|" << p[3]+0 << "]\033[0m" << std::endl;
  //   // printf("[HEADER]: [%d|%d|%d|%d]\n", p[0]+0, p[1]+0, p[2]+0, p[3]+0);
  // }

  global_state_->timeline.ActivityEndAll(entries);

  if (global_state_->rank != e.root_rank) {
    if (entries.size() > 1) {
      MemcpyOutFusionBuffer(data_ptr, entries, element_size);
    }
  }

  for (auto& name : response.tensor_names()) {
    global_state_->timestamp.Start(name, "EBCAST", std::to_string(e.batchid));
  }

  return Status::OK();
}

bool MPIBroadcast::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

MPIGatherBcast::MPIGatherBcast(MPIContext* mpi_context, HorovodGlobalState* global_state)
    : GatherBcastOp(global_state), mpi_context_(mpi_context) {}

bool MPIGatherBcast::Enabled(const ParameterManager& param_manager,
                             const std::vector<TensorTableEntry>& entries,
                             const Response& response) const {
  return true; //always use MPI gather and broadcast, because nccl has no gather premitive
}

Status MPIGatherBcast::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  assert(entries.size() == 1); //don't consider the fusion buffer
  auto e = entries[0];

  void *send_data, *recv_data, *data;

  if (global_state_->rank == e.root_rank) {
    recv_data = (void*) e.tensor->data();
  } else {
    recv_data = nullptr;
  }

  send_data = (void*) e.output->data();
  data = (void*) e.output->data();

  global_state_->timeline.ActivityStartAll(entries, MPI_GATHERBCAST);
  // gather compressed date from all workers
  int op = MPI_Gather(send_data, //send data
                      (int) e.output->shape().num_elements(), // send count
                      mpi_context_->GetMPIDataType(e.output->dtype()), // send data type
                      recv_data, //recv data
                      (int) e.output->shape().num_elements(), // recv count
                      mpi_context_->GetMPIDataType(e.output->dtype()), // recv data type
                      e.root_rank, // default is zero
                      mpi_context_->GetMPICommunicator(Communicator::GLOBAL));

  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Gather failed in GatherBcast execute, see MPI output for details.");
  }
  // decompression and reduce the gradients

  // root_rank compresses the reduced gradient

  // broadcast the compressed result to all workers
  op = MPI_Bcast(data, // data
                 (int) e.output->shape().num_elements(), // data count
                 mpi_context_->GetMPIDataType(e.output->dtype()), // data type
                 e.root_rank, // default is zero
                 mpi_context_->GetMPICommunicator(Communicator::GLOBAL));

  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Bcast failed in GatherBcast execute, see MPI output for details.");
  }

  global_state_->timeline.ActivityEndAll(entries);

  return Status::OK();
}

MPIGather::MPIGather(MPIContext* mpi_context, HorovodGlobalState* global_state)
    : GatherOp(global_state), mpi_context_(mpi_context) {}

bool MPIGather::Enabled(const ParameterManager& param_manager,
                             const std::vector<TensorTableEntry>& entries,
                             const Response& response) const {
  return true; //always use MPI gather, because nccl has no gather premitive
}

Status MPIGather::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {

  
  auto& e = entries[0];

  for (auto& name : response.tensor_names()) {
    global_state_->timestamp.Start(name, "BGATHER", std::to_string(e.batchid));
  }

  void *send_data = nullptr;
  void *recv_data = nullptr;

  int *recvcounts = nullptr;
  int *displcmnts = nullptr;

  int64_t **entry_component_sizes;
  int64_t **entry_component_offsets;

  int element_size = mpi_context_->GetMPITypeSize(e.tensor->dtype());
  int64_t total_num_elements = NumElements(entries);

  if (global_state_->rank == e.root_rank) { // allocate and initialize
    recvcounts = new int[global_state_->size]();
    displcmnts = new int[global_state_->size]();

    entry_component_sizes = new int64_t* [entries.size()];
    entry_component_offsets = new int64_t* [entries.size()];

    for (size_t ec = 0; ec < entries.size(); ec++) {
      entry_component_sizes[ec] = new int64_t [global_state_->size]();
      entry_component_offsets[ec] = new int64_t [global_state_->size]();
    }

    SetRecvcountsAndDisplcmnts(entries, response, entry_component_sizes, recvcounts, displcmnts);
    SetEntryComponentOffsets(entries, entry_component_sizes, recvcounts, entry_component_offsets);

    // root, copy the data to gathered place per entry
    // zero root has the data at offset zero
    // if (global_state_->rank != 0) {
    //   const auto &tensor_sizes = response.tensor_sizes();
    //   int64_t byte_size_of_entry = 0;
    //   for (size_t ec = 0; ec < entries.size(); ec++) {
    //     int64_t component_size = 0;
    //     for (int rc = 0; rc < e.root_rank; rc++) {
    //       component_size += tensor_sizes[ec * global_state_->size + rc];
    //     }
    //     uint8_t *move_data = (uint8_t *)entries[ec].output->data();
    //     byte_size_of_entry = entries[ec].num_elem > 0 ? entries[ec].num_elem*element_size : entries[ec].tensor->size();
    //     std::memcpy((void *)(move_data + component_size), entries[ec].tensor->data(), (size_t) byte_size_of_entry);
    //   }
    // }

    if (entries.size() > 1) {
      // recv_data = fusion_buffer
      auto& buffer = global_state_->fusion_buffer.GetBuffer(e.device, e.context->framework());
      recv_data = const_cast<void*>(buffer->AccessData(e.context));
    } else {
      recv_data = (void *) e.output->data();
    }
  } else {
    if (entries.size() > 1) {
      // copy at offset 0
      MemcpyInFusionBuffer(entries, send_data, element_size, 0);
    } else {
      send_data = (void *) e.tensor->data();
    }
  }

  global_state_->timeline.ActivityStartAll(entries, MPI_GATHER);
  // gather compressed date from all workers
  int op = MPI_Gatherv(send_data == nullptr ? MPI_IN_PLACE : send_data, //send data
                       (int) total_num_elements, // send count
                       mpi_context_->GetMPIDataType(e.tensor->dtype()), // send data type
                       recv_data, //recv data
                       recvcounts, // recv counts
                       displcmnts, // displacements
                       mpi_context_->GetMPIDataType(e.tensor->dtype()), // recv data type
                       e.root_rank, // default is zero
                       mpi_context_->GetMPICommunicator(Communicator::GLOBAL));

  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Gather failed in gatherv, see MPI output for details.");
  }

  int barrier_op = MPI_Barrier(mpi_context_->GetMPICommunicator(Communicator::GLOBAL));
  if (barrier_op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Barrier failed, see MPI output for details.");
  }

  global_state_->timeline.ActivityEndAll(entries);

  if (global_state_->rank == e.root_rank) {
    if (entries.size() > 1) {
      MemcpyOutFusionBuffer(entry_component_offsets, entry_component_sizes,
                            recv_data, element_size, entries);
    }
  }

  // if (global_state_->rank == e.root_rank) {
  //   std::cout << "root: ";
  //   for (auto& elem : entries) {
  //     int32_t header;
  //     void *src = (void *) ((uint8_t *)elem.output->data() + elem.num_elem);
  //     std::memcpy((void *)&header, src, 4);
  //     std::cout << elem.root_rank << ":" << elem.tensor_name << ":" << elem.num_elem << ":" << header << " ";
  //   }
  //   std::cout << std::endl;
  // } else {
  //   std::cout << "node: ";
  //   for (auto& elem : entries) {
  //     int32_t header;
  //     void *src = (void *)elem.tensor->data();
  //     std::memcpy((void *)&header, src, 4);
  //     std::cout << elem.root_rank << ":" << elem.tensor_name << ":" << elem.num_elem << ":" << header << " ";
  //   }
  //   std::cout << std::endl;
  // }

  if (global_state_->rank == e.root_rank) {
    delete [] recvcounts;
    delete [] displcmnts;

    for (size_t ec = 0; ec < entries.size(); ec++) {
      delete [] entry_component_sizes[ec];
      delete [] entry_component_offsets[ec];
    }
    delete [] entry_component_sizes;
    delete [] entry_component_offsets;
  }

  for (auto& name : response.tensor_names()) {
    global_state_->timestamp.Start(name, "EGATHER", std::to_string(e.batchid));
  }

  return Status::OK();
}

MPIAlltoall::MPIAlltoall(MPIContext* mpi_context, HorovodGlobalState* global_state)
    : AlltoallOp(global_state), mpi_context_(mpi_context) {}

bool MPIAlltoall::Enabled(const ParameterManager& param_manager,
                          const std::vector<TensorTableEntry>& entries,
                          const Response& response) const {
  return true; //always use MPI alltoall
}

Status MPIAlltoall::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  // construct a mpialltoall operator with the response and associated entries
  auto e = entries[0];
  void *send_data = nullptr;
  void *recv_data = nullptr;

  int *send_counts = nullptr;
  int *recv_counts = nullptr;
  int *send_displcmnts = nullptr;
  int *recv_displcmnts = nullptr;

  int64_t **entry_component_sizes;
  int64_t **entry_component_offsets;

  send_counts = new int[global_state_->size]();
  send_displcmnts = new int[global_state_->size]();
  recv_counts = new int[global_state_->size]();
  recv_displcmnts = new int[global_state_->size]();

  auto& timeline = global_state_->timeline;

  int element_size = mpi_context_->GetMPITypeSize(e.tensor->dtype());
  
  int roots_count = 0;
  bool recv_fusion = false;
  bool do_gather = false;
  int32_t entry_counts = 0;
  for (auto root_rank : response.root_ranks()) {
    if (global_state_->rank == root_rank) { // the same as gather operator
      do_gather = true;
      entry_counts = response.entries_counts()[roots_count];
      auto &first_entry = entries[response.entries_offsets()[roots_count]];

      entry_component_sizes = new int64_t* [entry_counts];
      entry_component_offsets = new int64_t* [entry_counts];

      for (size_t ec = 0; ec < entry_counts; ec++) {
        entry_component_sizes[ec] = new int64_t [global_state_->size]();
        entry_component_offsets[ec] = new int64_t [global_state_->size]();
      }

      SetRecvcountsAndDisplcmnts(entries, response, roots_count, entry_component_sizes, recv_counts, recv_displcmnts);
      SetEntryComponentOffsets(entries, entry_counts, entry_component_sizes, recv_counts, entry_component_offsets);
      
      recv_counts[global_state_->rank] = 0; // don't receive data from self
      // assign the recv_data
      if (entry_counts > 1) { // fusion
        recv_fusion = true;
        auto& buffer = global_state_->fusion_buffer.GetBuffer(first_entry.device, first_entry.context->framework());
        recv_data = const_cast<void*>(buffer->AccessData(first_entry.context));
      } else { 
        recv_data = (void *) first_entry.output->data();
      }
    }
    // construct send_counts
    SetSendcounts(entries, response, root_rank, roots_count, send_counts);
    roots_count++;
  }
  // construct send_displcmnts and copy send data to send fusion buffer
  for (int i = 1; i < global_state_->size; i++) {
    send_displcmnts[i] = send_displcmnts[i-1] + send_counts[i-1];
  }
  send_counts[global_state_->rank] = 0; // don't send data to self
  // copy data to send fusion buffer
  timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
  MemcpyInFusionBuffer(entries, response, element_size, send_counts, send_displcmnts, send_data);
  timeline.ActivityEndAll(entries);


  timeline.ActivityStartAll(entries, MPI_ALLTOALL);
  // alltoall, execute a few gather operators concurrently
  int op = MPI_Alltoallv(send_data, //send data
                         send_counts, // send counts
                         send_displcmnts,
                         mpi_context_->GetMPIDataType(e.tensor->dtype()),
                         recv_data,
                         recv_counts,
                         recv_displcmnts,
                         mpi_context_->GetMPIDataType(e.tensor->dtype()),
                         mpi_context_->GetMPICommunicator(Communicator::GLOBAL));
  
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Alltoall failed, see MPI output for details.");
  }
  timeline.ActivityEndAll(entries);

  if (do_gather) {
    if (recv_fusion) {
      // copy received data from fusion buffer
      timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
      MemcpyOutFusionBuffer(entry_component_offsets, entry_component_sizes, recv_data, response, element_size, entries);
      timeline.ActivityEndAll(entries);
    }
    for (size_t ec = 0; ec < entry_counts; ec++) {
      delete [] entry_component_sizes[ec];
      delete [] entry_component_offsets[ec];
    }
    delete [] entry_component_sizes;
    delete [] entry_component_offsets;
  }

  delete [] send_counts;
  delete [] send_displcmnts;
  delete [] recv_counts;
  delete [] recv_displcmnts;

  return Status::OK();
}

MPIBAlltoall::MPIBAlltoall(MPIContext* mpi_context, HorovodGlobalState* global_state)
    : AllgatherOp(global_state), mpi_context_(mpi_context) {}

bool MPIBAlltoall::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return true; //always use MPI BAllgather
}

void MPIBAlltoall::CopyInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                       const Response& response, int element_size, int roots_count,
                                       void*& buffer_data, int64_t* total_num_elements) {
  auto &first_entry = entries[0];
  auto &buffer = global_state_->fusion_buffer.GetSendBuffer(first_entry.device, first_entry.context->framework());
  buffer_data = const_cast<void *>(buffer->AccessData(first_entry.context));

  int64_t offset = 0;
  int64_t byte_size_of_entry;

  auto start_offset = response.entries_offsets()[roots_count];
  int64_t numelements = 0;
  for (size_t i = 0; i < response.entries_counts()[roots_count]; i++) {
    auto &e = entries[start_offset+i];
    void *buffer_data_at_offset = (uint8_t *)buffer_data + offset;
    byte_size_of_entry = e.num_elem > 0 ? e.num_elem*element_size : e.tensor->size();
    numelements += e.num_elem > 0 ? e.num_elem : e.tensor->shape().num_elements();
    std::memcpy(buffer_data_at_offset, e.tensor->data(), (size_t)byte_size_of_entry);
    offset += byte_size_of_entry;
  }
  *total_num_elements = numelements;
}

void MPIBAlltoall::CopyOutFusionBuffer(const void* buffer_data, const Response& response,
                                        const int* recvcounts, const int* recvdisplcmnts,
                                        int element_size, std::vector<TensorTableEntry>& entries) {
  auto& root_ranks = response.root_ranks();
  for (int rootidx = 0; rootidx < root_ranks.size(); rootidx++) {
    int rootid = root_ranks[rootidx];
    if (rootid != global_state_->rank) { // don't recv data from self
      assert(recvcounts[rootid] > 0);
      auto start_offset = response.entries_offsets()[rootidx];
      int64_t offset = recvdisplcmnts[rootid];
      int64_t byte_size_of_entry;
      for (size_t ec = 0; ec < response.entries_counts()[rootidx]; ec++) {
        auto& e = entries[start_offset+ec];
        void* buffer_data_at_offset = (uint8_t *)buffer_data + offset;
        byte_size_of_entry = e.num_elem > 0 ? e.num_elem*element_size : e.tensor->size();
        std::memcpy((void*)((uint8_t*)e.output->data()), buffer_data_at_offset, (size_t)byte_size_of_entry);
        offset += byte_size_of_entry;
      }
    }
  }
}

void MPIBAlltoall::SetRecvcounts(const std::vector<TensorTableEntry>& entries,
                                  const Response& response, const int root_rank,
                                  const int roots_count, int*& recvcounts) {
  // set the recvcounts[root_rank]
  int num_elements = 0;
  auto start_offset = response.entries_offsets()[roots_count];
  for (size_t ec = 0; ec < response.entries_counts()[roots_count]; ec++) {
    auto& e = entries[start_offset+ec];
    num_elements += e.num_elem > 0 ? e.num_elem : e.tensor->shape().num_elements();
  }
  recvcounts[root_rank] = num_elements;
}

Status MPIBAlltoall::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  auto& e = entries[0];
  void *send_data = nullptr;
  void *recv_data = nullptr;
  int *recv_counts = nullptr;
  int *recv_displcmnts = nullptr;
  int *send_counts = nullptr;
  int *send_displcmnts = nullptr;

  recv_counts = new int[global_state_->size]();
  recv_displcmnts = new int[global_state_->size]();
  send_counts = new int[global_state_->size]();
  send_displcmnts = new int[global_state_->size]();

  auto& timeline = global_state_->timeline;


  int element_size = mpi_context_->GetMPITypeSize(e.tensor->dtype());
  int64_t total_num_elements = 0;

  int roots_count = 0;
  int32_t entry_counts = 0;
  timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
  for (auto root_rank : response.root_ranks()) { // the broadcast launcher
    if (global_state_->rank == root_rank) { // the same as broadcast
      entry_counts = response.entries_counts()[roots_count];
      auto &first_entry = entries[response.entries_offsets()[roots_count]];
      // set total_num_elements
      if (entry_counts > 1) {
        // copy the tensor to send fusion buffer
        CopyInFusionBuffer(entries, response, element_size, roots_count, send_data, &total_num_elements);
      } else {
        send_data = (void *) first_entry.tensor->data();
        total_num_elements = first_entry.num_elem > 0 ? first_entry.num_elem : first_entry.tensor->shape().num_elements();
      }
      for (int i = 0; i < global_state_->size; i++) {
        send_counts[i] = total_num_elements;
        send_displcmnts[i] = 0; // do broadcast
      }
    }
    SetRecvcounts(entries, response, root_rank, roots_count, recv_counts);
    roots_count++;
  }
  send_counts[global_state_->rank] = 0; // don't send data to self
  recv_counts[global_state_->rank] = 0; // don't recv data from self
  // construct recv displacements
  for (int i = 1; i < global_state_->size; i++) {
    recv_displcmnts[i] = recv_displcmnts[i-1] + recv_counts[i-1];
  }

  auto& buffer = global_state_->fusion_buffer.GetBuffer(e.device, e.context->framework());
  recv_data = const_cast<void *>(buffer->AccessData(e.context));
  timeline.ActivityEndAll(entries);

  // int op = MPI_Allgatherv(send_data,
  //                         (int) total_num_elements,
  //                         mpi_context_->GetMPIDataType(e.tensor->dtype()),
  //                         recv_data,
  //                         recv_counts,
  //                         recv_displcmnts,
  //                         mpi_context_->GetMPIDataType(e.tensor->dtype()),
  //                         mpi_context_->GetMPICommunicator(Communicator::GLOBAL));
  timeline.ActivityStartAll(entries, MPI_ALLTOALL);
  int op = MPI_Alltoallv(send_data,
                         send_counts,
                         send_displcmnts,
                         mpi_context_->GetMPIDataType(e.tensor->dtype()),
                         recv_data,
                         recv_counts,
                         recv_displcmnts,
                         mpi_context_->GetMPIDataType(e.tensor->dtype()),
                         mpi_context_->GetMPICommunicator(Communicator::GLOBAL));
  
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI alltoallv failed in broadcast alltoall.");
  }
  timeline.ActivityEndAll(entries);

  timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
  CopyOutFusionBuffer(recv_data, response, recv_counts, recv_displcmnts, element_size, entries);
  timeline.ActivityEndAll(entries);

  delete [] recv_counts;
  delete [] recv_displcmnts;
  delete [] send_counts;
  delete [] send_displcmnts;
  return Status::OK();
}

} // namespace common
} // namespace horovod
