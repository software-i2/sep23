// Copyright by BeeX [2026]
#pragma once

#include <ros/param.h>

#include <set>
#include <string>
#include <vector>

namespace sep23 {

// Strict reads from one namespace: nothing is defaulted and every problem is collected.
class Params {
public:
    // `owned` are the sub-namespaces this node answers for; unread keys under them are reported.
    Params(std::string ns, std::vector<std::string> owned) : ns_(std::move(ns)), owned_(std::move(owned)) {}

    double number(const std::string &key) {
        XmlRpc::XmlRpcValue v;
        double              out = 0.0;
        if (fetch(key, v) && !toNumber(v, out)) {
            fail(key, "a number");
        }
        return out;
    }

    int whole(const std::string &key) {
        XmlRpc::XmlRpcValue v;
        if (!fetch(key, v)) {
            return 0;
        }
        if (v.getType() != XmlRpc::XmlRpcValue::TypeInt) {
            fail(key, "a whole number");
            return 0;
        }
        return static_cast<int>(v);
    }

    bool flag(const std::string &key) {
        XmlRpc::XmlRpcValue v;
        if (!fetch(key, v)) {
            return false;
        }
        if (v.getType() != XmlRpc::XmlRpcValue::TypeBoolean) {
            fail(key, "true or false");
            return false;
        }
        return static_cast<bool>(v);
    }

    std::string text(const std::string &key) {
        XmlRpc::XmlRpcValue v;
        if (!fetch(key, v)) {
            return {};
        }
        if (v.getType() != XmlRpc::XmlRpcValue::TypeString) {
            fail(key, "text");
            return {};
        }
        return static_cast<std::string>(v);
    }

    std::vector<double> numbers(const std::string &key, size_t count) {
        XmlRpc::XmlRpcValue v;
        std::vector<double> out(count, 0.0);
        if (fetch(key, v) && !toNumbers(v, count, out)) {
            fail(key, "a list of " + std::to_string(count) + " numbers");
        }
        return out;
    }

    std::vector<std::vector<double>> rows(const std::string &key, size_t columns) {
        XmlRpc::XmlRpcValue              v;
        std::vector<std::vector<double>> out;
        if (!fetch(key, v)) {
            return out;
        }
        for (int i = 0; v.getType() == XmlRpc::XmlRpcValue::TypeArray && i < v.size(); ++i) {
            std::vector<double> row(columns, 0.0);
            if (!toNumbers(v[i], columns, row)) {
                break;
            }
            out.push_back(row);
        }
        if (out.empty() || static_cast<int>(out.size()) != v.size()) {
            fail(key, "rows of " + std::to_string(columns) + " numbers");
            out.clear();
        }
        return out;
    }

    void require(bool ok, const std::string &key, const std::string &rule) {
        if (!ok) {
            errors_.push_back(ns_ + "/" + key + " must be " + rule);
        }
    }

    // Failed reads and broken rules, then keys nothing read under the owned sub-namespaces.
    std::string problems() const {
        std::string out;
        for (const std::string &line : errors_) {
            out += "  " + line + "\n";
        }
        std::vector<std::string> names;
        ros::param::getParamNames(names);
        for (const std::string &name : names) {
            for (const std::string &sub : owned_) {
                const std::string prefix = sub.empty() ? ns_ + "/" : ns_ + "/" + sub + "/";
                if (name.compare(0, prefix.size(), prefix) == 0 && read_.count(name) == 0) {
                    out += "  " + name + " is set but nothing reads it\n";
                    break;
                }
            }
        }
        return out;
    }

private:
    bool fetch(const std::string &key, XmlRpc::XmlRpcValue &v) {
        const std::string name = ns_ + "/" + key;
        read_.insert(name);
        if (!ros::param::get(name, v)) {
            errors_.push_back(name + " is missing");
            return false;
        }
        return true;
    }

    void fail(const std::string &key, const std::string &expected) { errors_.push_back(ns_ + "/" + key + " must be " + expected); }

    static bool toNumber(XmlRpc::XmlRpcValue &v, double &out) {
        if (v.getType() == XmlRpc::XmlRpcValue::TypeDouble) {
            out = static_cast<double>(v);
            return true;
        }
        if (v.getType() == XmlRpc::XmlRpcValue::TypeInt) {
            out = static_cast<int>(v);
            return true;
        }
        return false;
    }

    static bool toNumbers(XmlRpc::XmlRpcValue &v, size_t count, std::vector<double> &out) {
        if (v.getType() != XmlRpc::XmlRpcValue::TypeArray || static_cast<size_t>(v.size()) != count) {
            return false;
        }
        for (int i = 0; i < v.size(); ++i) {
            if (!toNumber(v[i], out[i])) {
                return false;
            }
        }
        return true;
    }

    std::string              ns_;
    std::vector<std::string> owned_;
    std::set<std::string>    read_;
    std::vector<std::string> errors_;
};

}  // namespace sep23
