#ifndef HOROVOD_TIMESTAMP_H
#define HOROVOD_TIMESTAMP_H

#include <atomic>
#include <boost/lockfree/spsc_queue.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "message.h"

namespace horovod {
namespace common {

struct TimestampRecord {
  std::string tensor_name;
  std::string op_name;
  std::string args;
  long ts_micros;
};

class TimestampWriter {
public:
  void Initialize(std::string file_name);
  inline bool IsHealthy() const { return healthy_; }
  void EnqueueWrite(const std::string& tensor_name,
                         const std::string& op_name,
                         const std::string& args,
                         long ts_micros);

private:
  void DoWrite(const TimestampRecord& r);
  void WriterLoop();

  // Are we healthy? the file can be written
  std::atomic_bool healthy_{false};

  // Timeline file.
  std::ofstream file_;

  // Timeline record queue.
  boost::lockfree::spsc_queue<TimestampRecord,
                              boost::lockfree::capacity<1048576>>
      record_queue_;

  // Mapping of tensor names to indexes. It is used to reduce size of the
  // timeline file.
  std::unordered_map<std::string, int> tensor_table_;
};

class Timestamp {
public:
  void Initialize(std::string file_name,
                  unsigned int horovod_size,
                  unsigned int horovod_rank);
  inline bool Initialized() const { return initialized_; }
  void Start(const std::string& tensor_name,
             const std::string& op_name = "",
             const std::string& args = "");

private:
  long TimeSinceStartMicros() const;
  void WriteEvent(const std::string& tensor_name,
                  const std::string& op_name = "",
                  const std::string& args = "");

  // Boolean flag indicating whether Timeline was initialized (and thus should
  // be recorded).
  bool initialized_ = false;

  // Timeline writer.
  TimestampWriter writer_;

  // Time point when Horovod was started.
  std::chrono::steady_clock::time_point start_time_;

  // A mutex that guards timeline state from concurrent access.
  std::recursive_mutex mutex_;

  // Map of ranks to their string representations.
  // std::to_string() is very slow.
  std::vector<std::string> rank_strings_;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_TIMESTAMP_H
