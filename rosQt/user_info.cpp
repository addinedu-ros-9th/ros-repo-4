// user_info.cpp
#include "user_info.h"
#include <iostream>

std::string UserInfoManager::user_id = "";
UserInfo UserInfoManager::user_info = {"", "", ""};

void UserInfoManager::set_user_id(const std::string& id) {
    if (!id.empty()) {
        user_id = id;
    } else {
        std::cout << "No user id" << std::endl;
        user_id = "";
    }
}

void UserInfoManager::set_user_info(const UserInfo& info) {
    if (!info.name.empty()) {
        std::cout << "[DEBUG] set_user_info called: " << info.name << ", " << info.email << std::endl;
        user_info = info;
    } else {
        std::cout << "No user found with the given user_id." << std::endl;
        user_info = {"", "", ""};
    }
}

std::string UserInfoManager::get_user_id() {
    return user_id;
}

UserInfo UserInfoManager::get_user_info() {
    std::cout << "[DEBUG] get_user_info: " << user_info.name << std::endl;
    return user_info;
}

void UserInfoManager::reset_user() {
    user_id = "";
    user_info = {"", "", ""};
}