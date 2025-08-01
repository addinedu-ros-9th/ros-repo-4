// user_info.h
#ifndef USER_INFO_H
#define USER_INFO_H

#include <string>

struct UserInfo {
    std::string name;
    std::string email;
    std::string hospital_name;
};

class UserInfoManager {
public:
    static void set_user_id(const std::string& id);
    static void set_user_info(const UserInfo& info);
    static std::string get_user_id();
    static UserInfo get_user_info();
    static void reset_user();

private:
    static std::string user_id;
    static UserInfo user_info;
};

#endif // USER_INFO_H