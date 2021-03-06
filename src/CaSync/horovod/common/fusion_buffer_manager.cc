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

#include "fusion_buffer_manager.h"

namespace horovod {
namespace common {

Status FusionBufferManager::InitializeBuffer(int64_t threshold, int device, std::shared_ptr<OpContext> context,
                                             std::function<void()> on_start_init,
                                             std::function<void()> on_end_init) {
  auto& elem = tensor_fusion_buffers_[std::make_tuple(device, context->framework())];
  auto& send_elem = tensor_send_buffers_[std::make_tuple(device, context->framework())];
  auto& buffer = elem.first;
  auto& send_buffer = send_elem.first;
  int64_t& size = elem.second;
  int64_t& send_size = elem.second;
  if (size != threshold) {
    buffer.reset();
    size = 0;
  }
  if (send_size != threshold) {
    send_buffer.reset();
    send_size = 0;
  }

  if (buffer == nullptr && send_buffer == nullptr) {
    on_start_init();
    size = threshold;
    send_size = threshold;
    // Lazily allocate persistent buffer for Tensor Fusion and keep it
    // forever per device.
    Status status = context->AllocatePersistent(threshold, &buffer);
    Status send_status = context->AllocatePersistent(threshold, &send_buffer);
    on_end_init();
    assert(status.type() == send_status.type());
    return status;
  }

  return Status::OK();
}

std::shared_ptr<PersistentBuffer>& FusionBufferManager::GetBuffer(int device, Framework framework) {
  return tensor_fusion_buffers_[std::make_tuple(device, framework)].first;
}

std::shared_ptr<PersistentBuffer>& FusionBufferManager::GetSendBuffer(int device, Framework framework) {
  return tensor_send_buffers_[std::make_tuple(device, framework)].first;
}

} // namespace common
} // namespace horovod
