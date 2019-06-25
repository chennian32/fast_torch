#pragma once
#include <corecrt_io.h>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <Windows.h>
class file_util{
public:
	file_util() {}
	~file_util() {}
	static std::string get_dir_name(std::string file_name){
		char drive[_MAX_DRIVE];
		char dir[_MAX_DIR];
		char fname[_MAX_FNAME];
		char ext[_MAX_EXT];
		_splitpath(file_name.c_str(), drive, dir, fname, ext);
		return dir;
	}
	static std::string get_file_name(std::string file_name){
		char drive[_MAX_DRIVE];
		char dir[_MAX_DIR];
		char fname[_MAX_FNAME];
		char ext[_MAX_EXT];
		_splitpath(file_name.c_str(), drive, dir, fname, ext);
		std::string fn = fname;
		fn += ext;
		return fn;
	}
	static std::string get_file_ext(std::string file_name){
		char drive[_MAX_DRIVE];
		char dir[_MAX_DIR];
		char fname[_MAX_FNAME];
		char ext[_MAX_EXT];
		_splitpath(file_name.c_str(), drive, dir, fname, ext);
		return ext;
	}
	static std::string get_file_name_without_ext(std::string file_name){
		char drive[_MAX_DRIVE];
		char dir[_MAX_DIR];
		char fname[_MAX_FNAME];
		char ext[_MAX_EXT];
		_splitpath(file_name.c_str(), drive, dir, fname, ext);
		return fname;
	}
	static bool is_dir(std::string file_name){
		struct stat s;
		if (stat(file_name.c_str(), &s) == 0) {
			if (s.st_mode&S_IFDIR)
				return true;
		}
		return false;
	}
	static bool is_file(std::string file_name){
		struct stat s;
		if (stat(file_name.c_str(), &s) == 0) {
			if (s.st_mode&S_IFREG)
				return true;
		}
		return false;
	}
	static std::vector<std::string> list_files_and_dirs(std::string root){
		std::vector<std::string> out;
		intptr_t handle;
		_finddata_t findData;
		std::string roots = root + "\\*.*";
		handle = _findfirst(roots.c_str(), &findData);    // 查找目录中的第一个文件
		if (handle == -1){
			std::cout << "Failed to find first file!\n";
			return out;
		}
		do{
			bool is_out = strcmp(findData.name, ".") == 0 || strcmp(findData.name, "..") == 0;
			if (is_out)
				continue;
			if (findData.attrib & (_A_NORMAL | _A_RDONLY | _A_SUBDIR|_A_ARCH))
				out.push_back(root+"\\"+findData.name);
		} while (_findnext(handle, &findData) == 0);    // 查找目录中的下一个文件

		std::cout << "Done!\n";
		_findclose(handle);    // 关闭搜索句柄
		return out;
	}
	static std::vector<std::string> list_dirs(std::string root){
		std::vector<std::string> out;
		intptr_t handle;
		_finddata_t findData;
		std::string roots = root + "\\*.*";
		handle = _findfirst(roots.c_str(), &findData);    // 查找目录中的第一个文件
		if (handle == -1){
			std::cout << "Failed to find first file!\n";
			return out;
		}

		do{
			bool is_out = strcmp(findData.name, ".") == 0 || strcmp(findData.name, "..") == 0;
			if (is_out)
				continue;
			if (findData.attrib & (_A_SUBDIR))
				out.push_back(root + "\\" + findData.name);
		} while (_findnext(handle, &findData) == 0);    // 查找目录中的下一个文件

		std::cout << "Done!\n";
		_findclose(handle);    // 关闭搜索句柄
		return out;
	}
	static std::vector<std::string> list_files(std::string root){
		std::vector<std::string> out;
		intptr_t handle;
		_finddata_t findData;
		std::string roots = root + "\\*.*";
		handle = _findfirst(roots.c_str(), &findData);    // 查找目录中的第一个文件
		if (handle == -1){
			std::cout << "Failed to find first file!\n";
			return out;
		}

		do{
			bool is_out = strcmp(findData.name, ".") == 0 || strcmp(findData.name, "..") == 0;
			if (is_out)
				continue;
			if (findData.attrib & (_A_NORMAL | _A_RDONLY|_A_ARCH))
				out.push_back(root + "\\" + findData.name);
		} while (_findnext(handle, &findData) == 0);    // 查找目录中的下一个文件

		std::cout << "Done!\n";
		_findclose(handle);    // 关闭搜索句柄
		return out;
	}
	static std::string to_local (const std::string & str){
		int nwLen = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, NULL, 0);

		wchar_t * pwBuf = new wchar_t[nwLen + 1];//一定要加1，不然会出现尾巴 
		memset(pwBuf, 0, nwLen * 2 + 2);

		MultiByteToWideChar(CP_UTF8, 0, str.c_str(), str.length(), pwBuf, nwLen);

		int nLen = WideCharToMultiByte(CP_ACP, 0, pwBuf, -1, NULL, NULL, NULL, NULL);

		char * pBuf = new char[nLen + 1];
		memset(pBuf, 0, nLen + 1);

		WideCharToMultiByte(CP_ACP, 0, pwBuf, nwLen, pBuf, nLen, NULL, NULL);

		std::string retStr = pBuf;

		delete[]pBuf;
		delete[]pwBuf;

		pBuf = NULL;
		pwBuf = NULL;

		return retStr;
	}
	static std::string to_utf8(const std::string & str){
		int nwLen = ::MultiByteToWideChar(CP_ACP, 0, str.c_str(), -1, NULL, 0);

		wchar_t * pwBuf = new wchar_t[nwLen + 1];//一定要加1，不然会出现尾巴 
		ZeroMemory(pwBuf, nwLen * 2 + 2);

		::MultiByteToWideChar(CP_ACP, 0, str.c_str(), str.length(), pwBuf, nwLen);

		int nLen = ::WideCharToMultiByte(CP_UTF8, 0, pwBuf, -1, NULL, NULL, NULL, NULL);

		char * pBuf = new char[nLen + 1];
		ZeroMemory(pBuf, nLen + 1);

		::WideCharToMultiByte(CP_UTF8, 0, pwBuf, nwLen, pBuf, nLen, NULL, NULL);

		std::string retStr(pBuf);

		delete[]pwBuf;
		delete[]pBuf;

		pwBuf = NULL;
		pBuf = NULL;

		return retStr;
	}
	static std::string read_file_text(std::string file_name){
		FILE* f = fopen(file_name.c_str(), "rt");
		fseek(f, 0, SEEK_END);
		int l = ftell(f);
		rewind(f);
		auto it = new char[l * sizeof(char) + 1];
		memset(it, 0, (l * sizeof(char) + 1));
		fread(it, sizeof(char), l, f);
		std::string read = it;
		delete[] it;
		fclose(f);
		return read;
	}
	static std::vector<std::string> split(const  std::string& s, const std::string& delim){
		std::vector<std::string> elems;
		size_t pos = 0;
		size_t len = s.length();
		size_t delim_len = delim.length();
		if (delim_len == 0) return elems;
		while (pos < len){
			int find_pos = s.find(delim, pos);
			if (find_pos < 0){
				elems.push_back(s.substr(pos, len - pos));
				break;
			}
			elems.push_back(s.substr(pos, find_pos - pos));
			pos = find_pos + delim_len;
		}
		return elems;
	}
	static std::vector<std::string> split_split (const  std::string& s, const std::string& delim,
		const std::string& delim2,int64_t index) {
		std::vector<std::string> elems;
		size_t pos = 0;
		size_t len = s.length();
		size_t delim_len = delim.length();
		if (delim_len == 0) return elems;
		while (pos < len) {
			int find_pos = s.find(delim, pos);
			if (find_pos < 0) {
				auto s1 = s.substr(pos, len - pos);
				auto s1s=split(s1, delim2);
				if(index<s1s.size()&&index>=0)
					elems.push_back(s1s[index]);
				break;
			}
			auto s2 = s.substr(pos, find_pos - pos);
			auto s2s = split(s2, delim2);
			if (index < s2s.size() && index >= 0)
				elems.push_back(s2s[index]);
			pos = find_pos + delim_len;
		}
		return elems;
	}
	static std::vector<int64_t> split_int64(const  std::string& s, const std::string& delim) {
		std::vector<int64_t> elems;
		size_t pos = 0;
		size_t len = s.length();
		size_t delim_len = delim.length();
		if (delim_len == 0) return elems;
		while (pos < len) {
			int find_pos = s.find(delim, pos);
			if (find_pos < 0) {
				int64_t i = atoll(s.substr(pos, len - pos).c_str());
				elems.push_back(i);
				break;
			}
			int64_t i = atoll(s.substr(pos, len - pos).c_str());
			elems.push_back(i);
			pos = find_pos + delim_len;
		}
		return elems;
	}
	static std::vector<int64_t> split_int64_sub_one(const  std::string& s, const std::string& delim) {
		std::vector<int64_t> elems;
		size_t pos = 0;
		size_t len = s.length();
		size_t delim_len = delim.length();
		if (delim_len == 0) return elems;
		while (pos < len) {
			int find_pos = s.find(delim, pos);
			if (find_pos < 0) {
				int64_t i = atoll(s.substr(pos, len - pos).c_str());
				elems.push_back(i-1);
				break;
			}
			int64_t i = atoll(s.substr(pos, len - pos).c_str());
			elems.push_back(i-1);
			pos = find_pos + delim_len;
		}
		return elems;
	}
};