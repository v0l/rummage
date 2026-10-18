#ifndef POW_EVENT_H
#define POW_EVENT_H

#include <stdint.h>
#include <string>
#include <vector>

struct PowEvent {
    std::string pubkey;
    uint64_t createdAt = 0;
    uint64_t kind = 0;
    std::vector<std::vector<std::string>> tags;
    std::string content;
};

bool parsePowEvent(const std::string &json, PowEvent &event, std::string &error);

std::string jsonString(const std::string &s);

std::string serializeTags(const std::vector<std::vector<std::string>> &tags);

std::string serializeEvent(const std::string &pubkey, uint64_t createdAt, uint64_t kind,
                           const std::vector<std::vector<std::string>> &tags, const std::string &content);

#endif
