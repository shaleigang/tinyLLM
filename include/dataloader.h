#pragma once

#include <vector>
#include <dirent.h>
#include <sys/stat.h>
#include <fstream>
#include <iostream>

#include "util.h"
#include "tensor.h"

namespace tllm {

class TinyStoriesLoader {
public:
    TinyStoriesLoader(string path, index_t batch, index_t len);
    index_t get_iter_len() { return iter_len_; }
    std::pair<Tensor, Tensor> next();

private:
    void getFiles(string path, std::vector<string>& files );
    int getFileSize(const char *fileName);

    string data_path_;
    std::vector<string> data_files_;
    index_t iter_len_;

    index_t file_id_;
    std::ifstream fins_;
    index_t file_iter_len_;

    index_t batch_size_;
    index_t max_seq_len_;
};






}