#ifndef FASTLOADER_H
#define FASTLOADER_H
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

class FastLoader {
public:
    float* data = nullptr;
    size_t fileSize = 0;
    size_t totalRows = 0;
    const size_t COLS = 448;

    bool load(const char* filename) {
        int fd = open(filename, O_RDONLY);
        if (fd == -1) return false;

        struct stat sb;
        fstat(fd, &sb);
        fileSize = sb.st_size;
        totalRows = fileSize / (COLS * sizeof(float));

        // MMAP: Mapping file to memory for zero-copy access
        data = (float*)mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
        
        // MLOCK: Prevent the OS from swapping this to disk for low-latency
        mlock(data, fileSize);

        close(fd);
        return data != MAP_FAILED;
    }

    ~FastLoader() {
        if (data && data != MAP_FAILED) munmap(data, fileSize);
    }
};
#endif
