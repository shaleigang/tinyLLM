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

TinyStoriesLoader::TinyStoriesLoader(string path, index_t batch, index_t len)
  : data_path_(path),
    batch_size_(batch),
    max_seq_len_(len),
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
        // std::cout << "file: " << data_path_ + data_files_[file_id_] << std::endl;
        // std::cout << "file len: " << getFileSize((data_path_ + data_files_[file_id_]).c_str()) << std::endl;
        // std::cout << "file_iter_len: " << file_iter_len_ << std::endl;
        // std::cout << "sum_iter_len: " << iter_len_ << std::endl;
    }
    // std::cout << file_iter_len_ << std::endl;
    --file_iter_len_;
    Tensor ret({batch_size_, max_seq_len_ - 1});
    Tensor label({batch_size_, max_seq_len_ - 1});
    unsigned short pos; 
    for (index_t b = 0; b < batch_size_; ++b) {
        for (index_t s = 0; s < max_seq_len_; ++s) {
            fins_.read(reinterpret_cast<char*>(&pos), sizeof(unsigned short));
            if (s != max_seq_len_ - 1) {
                ret[{b, s}] = pos;
            }
            if (s != 0) {
                label[{b, s - 1}] = pos;
            }
        }
    }
    
    ret.disable_grad();
    label.disable_grad();

    return {ret, label};
}
