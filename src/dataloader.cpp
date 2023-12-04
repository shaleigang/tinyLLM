#include "dataloader.h"

using namespace tllm;

void TinyStoriesLoader::getFiles(string path, std::vector<string>& files )
{
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(path.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            if (ent->d_type != DT_DIR)
                files.push_back(ent->d_name);
        }
        closedir(dir);
    } else {
        perror("");
    }
    return ;
}

int TinyStoriesLoader::getFileSize(const char *fileName) {

	if (fileName == NULL) {
		return 0;
	}
	
	struct stat statbuf;

	stat(fileName, &statbuf);
	
	int filesize = statbuf.st_size;

	return filesize;
}

TinyStoriesLoader::TinyStoriesLoader(string path, index_t batch, index_t len, index_t vocab_size)
  : data_path_(path),
    batch_size_(batch),
    max_seq_len_(len),
    vocab_size_(vocab_size),
    file_iter_len_(0),
    file_id_(-1) {
    
    getFiles(data_path_, data_files_);
    iter_len_ = 0;
    for (const auto& iter : data_files_) {
        size_t len = getFileSize((data_path_ + iter).c_str());
        iter_len_ += len / sizeof(unsigned short) / max_seq_len_ / batch_size_;
    }
}


std::pair<Tensor, Tensor> TinyStoriesLoader::next() {
    if (file_iter_len_ <= 0) {
        ++file_id_;
        file_iter_len_ = getFileSize((data_path_ + data_files_[file_id_]).c_str()) / sizeof(unsigned short) / max_seq_len_ / batch_size_;
        fins_.close();
        fins_.open(data_path_ + data_files_[file_id_], std::ifstream::binary);
        std::cout << "file: " << data_path_ + data_files_[file_id_] << std::endl;
        std::cout << "file len: " << getFileSize((data_path_ + data_files_[file_id_]).c_str()) << std::endl;
        std::cout << "file_iter_len: " << file_iter_len_ << std::endl;
        std::cout << "sum_iter_len: " << iter_len_ << std::endl;
    }
    // std::cout << file_iter_len_ << std::endl;
    --file_iter_len_;
    Tensor ret({batch_size_, max_seq_len_ - 1, vocab_size_});
    Tensor label({batch_size_, max_seq_len_ - 1});
    unsigned short pos; 
    for (int d = 0; d < batch_size_ * max_seq_len_; ++d) {
        fins_.read(reinterpret_cast<char*>(&pos), sizeof(unsigned short));
        std::cout << pos << std::endl;
        if (d % max_seq_len_ != (max_seq_len_ - 1))
            ret[d * vocab_size_ + pos] = 1;
        if (d > 0)
            label[d - 1] = pos;
    }
    ret.disable_grad();
    label.disable_grad();

    return {ret, label};
}
