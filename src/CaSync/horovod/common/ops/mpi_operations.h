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

#ifndef HOROVOD_MPI_OPERATIONS_H
#define HOROVOD_MPI_OPERATIONS_H

#include <iostream>

#include "mpi.h"

#include "../common.h"
#include "../global_state.h"
#include "../mpi_context.h"
#include "collective_operations.h"

namespace horovod {
namespace common {

class MPIAllreduce : public AllreduceOp {
public:
  MPIAllreduce(MPIContext* mpi_context, HorovodGlobalState* global_state);

  virtual ~MPIAllreduce() = default;

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  void MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                 const TensorTableEntry& e, void* buffer_data_at_offset) override;

  void MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                  const void* buffer_data_at_offset, TensorTableEntry& e) override;

  MPIContext* mpi_context_;
};

class MPIAllgather : public AllgatherOp {
public:
  MPIAllgather(MPIContext* mpi_context, HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  MPIContext* mpi_context_;
};

class MPIBAlltoall : public AllgatherOp {
public:
  MPIBAlltoall(MPIContext* mpi_context, HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  MPIContext* mpi_context_;

  void CopyInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                          const Response& response,
                          int element_size,
                          int roots_count,
                          void*& buffer_data,
                          int64_t* total_num_elements);

  void CopyOutFusionBuffer(const void* buffer_data,
                           const Response& response,
                           const int* recvcounts,
                           const int* recvdisplcmnts,
                           int element_size,
                           std::vector<TensorTableEntry>& entries);
                           
  void SetRecvcounts(const std::vector<TensorTableEntry>& entries,
                     const Response& response,
                     const int root_rank,
                     const int roots_count,
                     int*& recvcounts);
};

class MPIHierarchicalAllgather : public MPIAllgather {
public:
  MPIHierarchicalAllgather(MPIContext* mpi_context, HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

private:
  void Barrier();
};

class MPIBroadcast : public BroadcastOp {
public:
  MPIBroadcast(MPIContext* mpi_context, HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  MPIContext* mpi_context_;
};

class MPIGatherBcast : public GatherBcastOp {
public:
  MPIGatherBcast(MPIContext* mpi_context, HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  MPIContext* mpi_context_;
};

class MPIGather : public GatherOp {
public:
  MPIGather(MPIContext* mpi_context, HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  MPIContext* mpi_context_;
};

class MPIAlltoall : public AlltoallOp {
public:
  MPIAlltoall(MPIContext* mpi_context, HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  MPIContext* mpi_context_;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_MPI_OPERATIONS_H
