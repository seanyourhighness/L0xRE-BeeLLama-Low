#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>
#include <string>

struct ggml_tensor;

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes written so far
    virtual size_t n_bytes() = 0;
    virtual bool is_counting() const { return false; }
    // Device writers defer copies until destruction; retain temporary decoded
    // state tensors through that same copy boundary.
    virtual void write_tensor_owned(ggml_tensor * tensor,size_t offset,size_t size,std::shared_ptr<void> owner) {
        owned_tensors.push_back(std::move(owner));write_tensor(tensor,offset,size);
    }
    bool has_owned_tensors() const { return !owned_tensors.empty(); }

    void write_string(const std::string & str);
private:
    // Base members outlive every derived writer's destructor/copy boundary.
    std::vector<std::shared_ptr<void>> owned_tensors;
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // State restore is a transaction. Tensor writes and publication callbacks
    // are prepared while the frame is parsed, then made visible only after the
    // complete frame has validated. Destruction without commit cancels them.
    virtual void stage_tensor_set(ggml_tensor * tensor, const void * src, size_t offset, size_t size) = 0;
    virtual void stage_tensor_clear(ggml_tensor * tensor, size_t offset, size_t size) = 0;
    virtual void on_commit(std::function<void()> callback) = 0;
    virtual void commit() = 0;
    virtual void cancel() = 0;

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);
};
