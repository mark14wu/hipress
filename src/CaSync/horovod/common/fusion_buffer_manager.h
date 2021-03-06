// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

#ifndef HOROVOD_FUSION_BUFFER_MANAGER_H
#define HOROVOD_FUSION_BUFFER_MANAGER_H

#include <iostream>
#include <unordered_map>
#include <assert.h>

#include "common.h"
#include "hashes.h"
#include "operations.h"

// #undef NDEBUG

namespace horovod {
namespace common {

// Encapsulates the process of creating and destroying fusion buffers as the requested
// threshold is changed.
class FusionBufferManager {
public:
  // Initializes a buffer of the given threshold size if not already cached.
  //
  // Args:
  //  threshold: Size of the buffer in bytes.
  //  device: Device ID to associate the buffer.
  //  context: Framework used to create the buffer and associate it.
  //  on_start_init: Callback on starting buffer initialization.
  //  on_end_init: Callback on completing buffer initialization.
  Status InitializeBuffer(int64_t threshold,
                          int device, std::shared_ptr<OpContext> context,
                          std::function<void()> on_start_init,
                          std::function<void()> on_end_init);

  // Returns the buffer associated with the given device and framework, or null.
  std::shared_ptr<PersistentBuffer>& GetBuffer(int device, Framework framework);
  // Returns the send buffer associated with the given device and framework
  std::shared_ptr<PersistentBuffer>& GetSendBuffer(int device, Framework framework);

private:
  // Memory buffers for Tensor Fusion.  They are keyed off device ID and
  // framework, and all are allocated tensor_fusion_threshold bytes if
  // initialized.
  std::unordered_map<
      std::tuple<int, Framework>,
      std::pair<std::shared_ptr<PersistentBuffer>, int64_t>> tensor_fusion_buffers_;
  std::unordered_map<
      std::tuple<int, Framework>,
      std::pair<std::shared_ptr<PersistentBuffer>, int64_t>> tensor_send_buffers_;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_FUSION_BUFFER_MANAGER_H
