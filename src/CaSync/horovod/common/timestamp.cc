#include <cassert>
#include <chrono>
#include <sstream>
#include <thread>

#include "logging.h"
#include "timestamp.h"

namespace horovod {
namespace common {

void TimestampWriter::Initialize(std::string file_name) {
  file_.open(file_name, std::ios::out | std::ios::trunc);
  if (file_.good()) {
    healthy_ = true;
    // Spawn writer thread.
    std::thread writer_thread(&TimestampWriter::WriterLoop, this);
    writer_thread.detach();
  } else {
    LOG(ERROR) << "Error opening the Horovod timestamp file " << file_name
               << ", will not write a timestamp record.";
  }
}

void TimestampWriter::EnqueueWrite(const std::string& tensor_name,
                                   const std::string& op_name,
                                   const std::string& args,
                                   long ts_micros) {
  TimestampRecord r{};
  r.tensor_name = tensor_name;
  r.op_name = op_name;
  r.args = args;
  r.ts_micros = ts_micros;

  while (healthy_ && !record_queue_.push(r))
    ;
}

void TimestampWriter::DoWrite(const TimestampRecord& r) {
  auto& tensor_idx = tensor_table_[r.tensor_name];
  if (tensor_idx == 0) {
    tensor_idx = (int)tensor_table_.size();
  }

  std::string output = r.op_name + "," + r.tensor_name + "," + std::to_string(r.ts_micros);
  if (r.args != "") {
    output += "," + r.args;
  }
  file_ << output << std::endl;
}

void TimestampWriter::WriterLoop() {
  while (healthy_) {
    while (healthy_ && !record_queue_.empty()) {
      auto& r = record_queue_.front();
      DoWrite(r);
      record_queue_.pop();
      if (!file_.good()) {
        LOG(ERROR) << "Error writing to the Horovod Timeline after it was "
                      "successfully opened, will stop writing the timeline.";
        healthy_ = false;
      }
    }
    // Allow scheduler to schedule other work for this core.
    std::this_thread::yield();
  }
}

void Timestamp::Initialize(std::string file_name,
                           unsigned int horovod_size,
                           unsigned int horovod_rank) {
  if (initialized_) {
    return;
  }

  // Start the writer.
  file_name += std::to_string(horovod_rank) + ".log";
  writer_.Initialize(std::move(file_name));

  // Initialize if we were able to open the file successfully.
  initialized_ = writer_.IsHealthy();

  // Pre-initialize the string representation for each rank.
  rank_strings_ = std::vector<std::string>(horovod_size);
  for (unsigned int i = 0; i < horovod_size; i++) {
    rank_strings_[i] = std::to_string(i);
  }
}

long Timestamp::TimeSinceStartMicros() const {
  auto now = std::chrono::steady_clock::now();
  auto ts = now - start_time_;
  return std::chrono::duration_cast<std::chrono::microseconds>(ts).count();
}

// Write event to the Horovod Timestamp file.
void Timestamp::WriteEvent(const std::string& tensor_name,
                           const std::string& op_name,
                           const std::string& args) {
  auto ts_micros = TimeSinceStartMicros();
  writer_.EnqueueWrite(tensor_name, op_name, args, ts_micros);
}

void Timestamp::Start(const std::string& tensor_name,
                      const std::string& op_name,
                      const std::string& args) {
  if (!initialized_) {
    return;
  }

  std::lock_guard<std::recursive_mutex> guard(mutex_);
  WriteEvent(tensor_name, op_name, args);
}

} // namespace common
} // namespace horovod
