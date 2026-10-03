// ============================================================
//  xorzen.cpp — src/optimized/expert_mmap.cpp
//  Memory-mapped expert loading (zero-copy)
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
#include "xorzen/optimized/expert_mmap.h"

#ifdef XORZEN_ENABLE_MMAP

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#include <iostream>
#include <stdexcept>
#include <fstream>

namespace xorzen {
namespace optimized {

MMapRegion::~MMapRegion() {
    close();
}

MMapRegion::MMapRegion(MMapRegion&& other) noexcept
    : data_(other.data_), size_(other.size_) {
#ifdef _WIN32
    file_handle_ = other.file_handle_;
    map_handle_ = other.map_handle_;
    other.file_handle_ = nullptr;
    other.map_handle_ = nullptr;
#else
    fd_ = other.fd_;
    other.fd_ = -1;
#endif
    other.data_ = nullptr;
    other.size_ = 0;
}

MMapRegion& MMapRegion::operator=(MMapRegion&& other) noexcept {
    if (this != &other) {
        close();
        data_ = other.data_;
        size_ = other.size_;
#ifdef _WIN32
        file_handle_ = other.file_handle_;
        map_handle_ = other.map_handle_;
        other.file_handle_ = nullptr;
        other.map_handle_ = nullptr;
#else
        fd_ = other.fd_;
        other.fd_ = -1;
#endif
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

bool MMapRegion::open(const std::filesystem::path& path, bool read_only) {
    close();

#ifdef _WIN32
    DWORD access = read_only ? GENERIC_READ : (GENERIC_READ | GENERIC_WRITE);
    DWORD share = FILE_SHARE_READ;
    DWORD disposition = OPEN_EXISTING;

    file_handle_ = CreateFileW(path.c_str(), access, share, NULL, disposition, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file_handle_ == INVALID_HANDLE_VALUE) {
        file_handle_ = nullptr;
        return false;
    }

    LARGE_INTEGER fs;
    if (!GetFileSizeEx(file_handle_, &fs)) {
        CloseHandle(file_handle_);
        file_handle_ = nullptr;
        return false;
    }
    size_ = static_cast<size_t>(fs.QuadPart);

    DWORD prot = read_only ? PAGE_READONLY : PAGE_READWRITE;
    map_handle_ = CreateFileMappingW(file_handle_, NULL, prot, 0, 0, NULL);
    if (!map_handle_) {
        CloseHandle(file_handle_);
        file_handle_ = nullptr;
        return false;
    }

    DWORD map_access = read_only ? FILE_MAP_READ : FILE_MAP_WRITE;
    data_ = MapViewOfFile(map_handle_, map_access, 0, 0, 0);
    if (!data_) {
        CloseHandle(map_handle_);
        CloseHandle(file_handle_);
        map_handle_ = nullptr;
        file_handle_ = nullptr;
        return false;
    }

#else
    int flags = read_only ? O_RDONLY : O_RDWR;
    fd_ = ::open(path.c_str(), flags);
    if (fd_ == -1) return false;

    struct stat st;
    if (fstat(fd_, &st) == -1) {
        ::close(fd_);
        fd_ = -1;
        return false;
    }
    size_ = st.st_size;

    int prot = read_only ? PROT_READ : (PROT_READ | PROT_WRITE);
    data_ = mmap(NULL, size_, prot, MAP_SHARED, fd_, 0);
    if (data_ == MAP_FAILED) {
        data_ = nullptr;
        ::close(fd_);
        fd_ = -1;
        return false;
    }
#endif
    return true;
}

void MMapRegion::close() {
    if (!data_) return;

#ifdef _WIN32
    UnmapViewOfFile(data_);
    CloseHandle(map_handle_);
    CloseHandle(file_handle_);
    map_handle_ = nullptr;
    file_handle_ = nullptr;
#else
    munmap(data_, size_);
    ::close(fd_);
    fd_ = -1;
#endif
    data_ = nullptr;
    size_ = 0;
}

void MMapRegion::prefetch(size_t offset, size_t length) {
    if (!data_) return;
    if (length == 0) length = size_ - offset;
    
#ifdef _WIN32
    // Windows doesn't have a direct equivalent to madvise(MADV_WILLNEED) 
    // but we can use PrefetchVirtualMemory on Windows 8+
    // For now, a simple loop touching pages is a portable way to "warm up" the cache
    volatile char* p = static_cast<volatile char*>(data_) + offset;
    for (size_t i = 0; i < length; i += 4096) {
        (void)p[i];
    }
#else
    madvise(static_cast<char*>(data_) + offset, length, MADV_WILLNEED);
#endif
}

torch::Tensor tensor_from_mmap(
    MMapRegion& region,
    torch::ScalarType dtype,
    torch::IntArrayRef sizes,
    size_t offset) 
{
    if (!region.is_open()) {
        throw std::runtime_error("MMapRegion is not open");
    }

    void* ptr = static_cast<char*>(region.data()) + offset;
    
    // Create a tensor that views this memory
    // Note: We don't own the memory, so the deleter does nothing
    auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
    return torch::from_blob(ptr, sizes, options);
}

void save_expert_mmap_file(
    const std::filesystem::path& path,
    const std::vector<torch::Tensor>& weights,
    int64_t hidden_dim,
    int64_t intermediate_dim,
    int quantization_type) 
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("Failed to open file for writing: " + path.string());

    ExpertFileHeader header;
    header.magic = ExpertFileHeader::MAGIC;
    header.version = ExpertFileHeader::VERSION;
    header.hidden_dim = hidden_dim;
    header.intermediate_dim = intermediate_dim;
    header.quantization_type = quantization_type;
    
    size_t total_weight_bytes = 0;
    for (const auto& w : weights) {
        total_weight_bytes += w.nbytes();
    }
    header.weight_bytes = total_weight_bytes;

    ofs.write(reinterpret_cast<const char*>(&header), sizeof(header));
    
    for (const auto& w : weights) {
        auto wc = w.to(torch::kCPU).contiguous();
        ofs.write(reinterpret_cast<const char*>(wc.data_ptr()), wc.nbytes());
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> load_expert_mmap_file(
    MMapRegion& region,
    torch::Device device) 
{
    if (!region.is_open()) throw std::runtime_error("MMapRegion not open");
    if (region.size() < sizeof(ExpertFileHeader)) throw std::runtime_error("File too small for header");

    const auto* header = static_cast<const ExpertFileHeader*>(region.data());
    if (!header->is_valid()) throw std::runtime_error("Invalid expert file header");

    size_t offset = ExpertFileHeader::HEADER_SIZE;
    
    // Assuming 3 tensors: gate, up, down (standard Llama FFN)
    // Shapes: [intermediate_dim, hidden_dim], [intermediate_dim, hidden_dim], [hidden_dim, intermediate_dim]
    
    torch::ScalarType dtype = (header->quantization_type == 0) ? torch::kFloat32 : torch::kFloat16;
    
    auto gate = tensor_from_mmap(region, dtype, { (int64_t)header->intermediate_dim, (int64_t)header->hidden_dim }, offset);
    offset += gate.nbytes();
    
    auto up = tensor_from_mmap(region, dtype, { (int64_t)header->intermediate_dim, (int64_t)header->hidden_dim }, offset);
    offset += up.nbytes();
    
    auto down = tensor_from_mmap(region, dtype, { (int64_t)header->hidden_dim, (int64_t)header->intermediate_dim }, offset);
    
    // If device is not CPU, we must copy
    if (!device.is_cpu()) {
        return { gate.to(device), up.to(device), down.to(device) };
    }
    
    return { gate, up, down };
}

} // namespace optimized
} // namespace xorzen

#endif // XORZEN_ENABLE_MMAP
