#include "PowEvent.h"

#include <cstdio>

namespace {

struct Parser {
    const std::string &s;
    size_t i = 0;
    std::string error;

    explicit Parser(const std::string &src) : s(src) {}

    bool fail(const std::string &message) {
        if (error.empty()) error = message + " at byte " + std::to_string(i);
        return false;
    }

    void ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++;
    }

    bool eat(char c) {
        ws();
        if (i < s.size() && s[i] == c) {
            i++;
            return true;
        }
        return false;
    }

    bool hex4(uint32_t &v) {
        if (i + 4 > s.size()) return fail("truncated \\u escape");
        v = 0;
        for (int k = 0; k < 4; k++) {
            char c = s[i++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else return fail("bad \\u escape");
        }
        return true;
    }

    static void utf8(std::string &out, uint32_t cp) {
        if (cp < 0x80) {
            out += (char)cp;
        } else if (cp < 0x800) {
            out += (char)(0xC0 | (cp >> 6));
            out += (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += (char)(0xE0 | (cp >> 12));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        } else {
            out += (char)(0xF0 | (cp >> 18));
            out += (char)(0x80 | ((cp >> 12) & 0x3F));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        }
    }

    bool string(std::string &out) {
        ws();
        if (i >= s.size() || s[i] != '"') return fail("expected string");
        i++;
        out.clear();
        while (i < s.size()) {
            unsigned char c = (unsigned char)s[i++];
            if (c == '"') return true;
            if (c < 0x20) return fail("control character in string");
            if (c != '\\') {
                out += (char)c;
                continue;
            }
            if (i >= s.size()) break;
            char e = s[i++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp;
                    if (!hex4(cp)) return false;
                    if (cp >= 0xD800 && cp < 0xDC00) {
                        uint32_t lo;
                        if (i + 2 > s.size() || s[i] != '\\' || s[i + 1] != 'u') return fail("unpaired surrogate");
                        i += 2;
                        if (!hex4(lo)) return false;
                        if (lo < 0xDC00 || lo >= 0xE000) return fail("unpaired surrogate");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (cp >= 0xDC00 && cp < 0xE000) {
                        return fail("unpaired surrogate");
                    }
                    utf8(out, cp);
                    break;
                }
                default: return fail("bad escape");
            }
        }
        return fail("unterminated string");
    }

    bool uint(uint64_t &out) {
        ws();
        size_t start = i;
        out = 0;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
            uint64_t d = (uint64_t)(s[i] - '0');
            if (out > (UINT64_MAX - d) / 10) return fail("number too large");
            out = out * 10 + d;
            i++;
        }
        if (i == start) return fail("expected non-negative integer");
        if (i < s.size() && (s[i] == '.' || s[i] == 'e' || s[i] == 'E')) return fail("expected integer");
        return true;
    }

    bool skip() {
        ws();
        if (i >= s.size()) return fail("unexpected end of input");
        char c = s[i];
        if (c == '"') {
            std::string tmp;
            return string(tmp);
        }
        if (c == '{' || c == '[') {
            char close = c == '{' ? '}' : ']';
            i++;
            if (eat(close)) return true;
            do {
                if (c == '{') {
                    std::string key;
                    if (!string(key) || !eat(':')) return fail("bad object");
                }
                if (!skip()) return false;
            } while (eat(','));
            return eat(close) || fail("unterminated container");
        }
        size_t start = i;
        while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']' && s[i] != ' ' && s[i] != '\n' &&
               s[i] != '\t' && s[i] != '\r') i++;
        std::string lit = s.substr(start, i - start);
        if (lit.empty()) return fail("unexpected character");
        if (lit == "true" || lit == "false" || lit == "null") return true;
        for (char ch : lit) {
            if (!((ch >= '0' && ch <= '9') || ch == '-' || ch == '+' || ch == '.' || ch == 'e' || ch == 'E')) {
                return fail("bad literal");
            }
        }
        return true;
    }

    bool tags(std::vector<std::vector<std::string>> &out) {
        if (!eat('[')) return fail("tags must be an array");
        if (eat(']')) return true;
        do {
            if (!eat('[')) return fail("every tag must be an array");
            std::vector<std::string> tag;
            if (!eat(']')) {
                do {
                    std::string v;
                    if (!string(v)) return fail("tag values must be strings");
                    tag.push_back(v);
                } while (eat(','));
                if (!eat(']')) return fail("unterminated tag");
            }
            if (tag.empty()) return fail("empty tag");
            out.push_back(tag);
        } while (eat(','));
        return eat(']') || fail("unterminated tags");
    }
};

}

bool parsePowEvent(const std::string &json, PowEvent &event, std::string &error) {
    Parser p(json);
    bool havePubkey = false, haveCreatedAt = false, haveKind = false;
    std::vector<std::vector<std::string>> tags;
    if (!p.eat('{')) {
        error = "event must be a JSON object";
        return false;
    }
    if (!p.eat('}')) {
        do {
            std::string key;
            if (!p.string(key) || !p.eat(':')) {
                error = p.error.empty() ? "bad object" : p.error;
                return false;
            }
            p.ws();
            bool isNull = json.compare(p.i, 4, "null") == 0;
            bool ok;
            if (key == "pubkey" && !isNull) ok = p.string(event.pubkey), havePubkey = ok;
            else if (key == "created_at") ok = p.uint(event.createdAt), haveCreatedAt = ok;
            else if (key == "kind") ok = p.uint(event.kind), haveKind = ok;
            else if (key == "content" && !isNull) ok = p.string(event.content);
            else if (key == "tags" && !isNull) ok = p.tags(tags);
            else ok = p.skip();
            if (!ok) {
                error = key + ": " + p.error;
                return false;
            }
        } while (p.eat(','));
        if (!p.eat('}')) {
            error = p.error.empty() ? "unterminated object" : p.error;
            return false;
        }
    }
    p.ws();
    if (p.i != json.size()) {
        error = "trailing data after event";
        return false;
    }
    if (!havePubkey || !haveCreatedAt || !haveKind) {
        error = "event must contain pubkey, created_at and kind";
        return false;
    }
    if (event.kind > 65535) {
        error = "kind must be between 0 and 65535";
        return false;
    }
    event.tags.clear();
    for (auto &t : tags) {
        if (t[0] != "nonce") event.tags.push_back(t);
    }
    return true;
}

std::string jsonString(const std::string &s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char)c;
                }
        }
    }
    return out + "\"";
}

std::string serializeTags(const std::vector<std::vector<std::string>> &tags) {
    std::string out = "[";
    for (size_t t = 0; t < tags.size(); t++) {
        if (t) out += ',';
        out += '[';
        for (size_t v = 0; v < tags[t].size(); v++) {
            if (v) out += ',';
            out += jsonString(tags[t][v]);
        }
        out += ']';
    }
    return out + "]";
}

std::string serializeEvent(const std::string &pubkey, uint64_t createdAt, uint64_t kind,
                           const std::vector<std::vector<std::string>> &tags, const std::string &content) {
    return "[0," + jsonString(pubkey) + "," + std::to_string(createdAt) + "," + std::to_string(kind) + "," +
           serializeTags(tags) + "," + jsonString(content) + "]";
}
