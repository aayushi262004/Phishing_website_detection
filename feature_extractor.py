# feature_extractor.py
# Strict/full feature extractor to match feature_groups.pkl
# Requires: requests, tldextract, python-whois, dnspython
# Optional (better ASN): ipwhois
import re
import socket
import ssl
from datetime import datetime
from urllib.parse import urlparse, parse_qs

import dns.resolver
import pandas as pd
import requests
import tldextract

# Try optional ipwhois for ASN lookup
try:
    from ipwhois import IPWhois
    _HAS_IPWHOIS = True
except Exception:
    _HAS_IPWHOIS = False

# -------------------------
# Helper utilities
# -------------------------
SHORTENERS = ["bit.ly", "goo.gl", "tinyurl", "t.co", "ow.ly", "is.gd", "buff.ly", "adf.ly"]

def safe_get(obj, key, default=None):
    try:
        return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
    except Exception:
        return default

def extract_domain(url):
    try:
        ext = tldextract.extract(url)
        if ext.suffix:
            return ext.domain + "." + ext.suffix
        return ext.domain
    except Exception:
        return ""

def count_chars(text, ch):
    try:
        return text.count(ch)
    except Exception:
        return 0

def is_ip_string(s):
    try:
        socket.inet_aton(s)
        return True
    except Exception:
        return False

# -------------------------
# URL-level features (url + domain counts)
# -------------------------
def extract_url_features(url):
    parsed = urlparse(url if url.startswith(("http://", "https://")) else "http://" + url)
    full_url = url if url.startswith(("http://", "https://")) else "http://" + url
    domain = extract_domain(full_url)

    features = {}

    # char counts on full URL
    features["qty_dot_url"] = count_chars(full_url, ".")
    features["qty_hyphen_url"] = count_chars(full_url, "-")
    features["qty_underline_url"] = count_chars(full_url, "_")
    features["qty_slash_url"] = count_chars(full_url, "/")
    features["qty_questionmark_url"] = count_chars(full_url, "?")
    features["qty_equal_url"] = count_chars(full_url, "=")
    features["qty_at_url"] = count_chars(full_url, "@")
    features["qty_and_url"] = count_chars(full_url, "&")
    features["qty_exclamation_url"] = count_chars(full_url, "!")
    features["qty_space_url"] = count_chars(full_url, " ")
    features["qty_tilde_url"] = count_chars(full_url, "~")
    features["qty_comma_url"] = count_chars(full_url, ",")
    features["qty_plus_url"] = count_chars(full_url, "+")
    features["qty_asterisk_url"] = count_chars(full_url, "*")
    features["qty_hashtag_url"] = count_chars(full_url, "#")
    features["qty_dollar_url"] = count_chars(full_url, "$")
    features["qty_percent_url"] = count_chars(full_url, "%")
    # tld presence in the end of url (approx)
    features["qty_tld_url"] = 1 if parsed.path.endswith("." + (tldextract.extract(full_url).suffix or "")) else 0
    features["length_url"] = len(full_url)

    # Domain-level char counts (domain string)
    domain_str = domain or ""
    features["qty_dot_domain"] = count_chars(domain_str, ".")
    features["qty_hyphen_domain"] = count_chars(domain_str, "-")
    features["qty_underline_domain"] = count_chars(domain_str, "_")
    features["qty_slash_domain"] = count_chars(domain_str, "/")   # usually 0 in domain
    features["qty_questionmark_domain"] = count_chars(domain_str, "?")
    features["qty_equal_domain"] = count_chars(domain_str, "=")
    features["qty_at_domain"] = count_chars(domain_str, "@")
    features["qty_and_domain"] = count_chars(domain_str, "&")
    features["qty_exclamation_domain"] = count_chars(domain_str, "!")
    features["qty_space_domain"] = count_chars(domain_str, " ")
    features["qty_tilde_domain"] = count_chars(domain_str, "~")
    features["qty_comma_domain"] = count_chars(domain_str, ",")
    features["qty_plus_domain"] = count_chars(domain_str, "+")
    features["qty_asterisk_domain"] = count_chars(domain_str, "*")
    features["qty_hashtag_domain"] = count_chars(domain_str, "#")
    features["qty_dollar_domain"] = count_chars(domain_str, "$")
    features["qty_percent_domain"] = count_chars(domain_str, "%")
    # vowels in domain
    features["qty_vowels_domain"] = sum(1 for c in domain_str.lower() if c in "aeiou")
    # server_client_domain: heuristic -> 1 if domain looks like server/client keywords present
    features["server_client_domain"] = 1 if any(x in domain_str.lower() for x in ["server", "client", "gateway", "admin", "mail"]) else 0

    # domain_length and domain_in_ip will be handled in DOM extractor (they belong to dom_features in feature_groups)

    # time_domain_activation/time_domain_expiration included here because they are also listed in url_features
    # But actual values will be computed in network extractor to avoid duplication; set None here, will be filled later
    features["time_domain_activation"] = None
    features["time_domain_expiration"] = None

    return features

# -------------------------
# DOM / path / file / params features
# -------------------------
def extract_dom_features(url):
    parsed = urlparse(url if url.startswith(("http://", "https://")) else "http://" + url)
    path = parsed.path or ""
    query = parsed.query or ""
    file_part = path.split("/")[-1] if path else ""
    directories = "/".join([p for p in path.split("/")[:-1] if p != ""])

    features = {}
    # directory counts
    features["qty_dot_directory"] = count_chars(directories, ".")
    features["qty_hyphen_directory"] = count_chars(directories, "-")
    features["qty_underline_directory"] = count_chars(directories, "_")
    features["qty_slash_directory"] = count_chars(directories, "/")
    features["qty_questionmark_directory"] = count_chars(directories, "?")
    features["qty_equal_directory"] = count_chars(directories, "=")
    features["qty_at_directory"] = count_chars(directories, "@")
    features["qty_and_directory"] = count_chars(directories, "&")
    features["qty_exclamation_directory"] = count_chars(directories, "!")
    features["qty_space_directory"] = count_chars(directories, " ")
    features["qty_tilde_directory"] = count_chars(directories, "~")
    features["qty_comma_directory"] = count_chars(directories, ",")
    features["qty_plus_directory"] = count_chars(directories, "+")
    features["qty_asterisk_directory"] = count_chars(directories, "*")
    features["qty_hashtag_directory"] = count_chars(directories, "#")
    features["qty_dollar_directory"] = count_chars(directories, "$")
    features["qty_percent_directory"] = count_chars(directories, "%")
    features["directory_length"] = len(directories)

    # file counts
    features["qty_dot_file"] = count_chars(file_part, ".")
    features["qty_hyphen_file"] = count_chars(file_part, "-")
    features["qty_underline_file"] = count_chars(file_part, "_")
    features["qty_slash_file"] = count_chars(file_part, "/")
    features["qty_questionmark_file"] = count_chars(file_part, "?")
    features["qty_equal_file"] = count_chars(file_part, "=")
    features["qty_at_file"] = count_chars(file_part, "@")
    features["qty_and_file"] = count_chars(file_part, "&")
    features["qty_exclamation_file"] = count_chars(file_part, "!")
    features["qty_space_file"] = count_chars(file_part, " ")
    features["qty_tilde_file"] = count_chars(file_part, "~")
    features["qty_comma_file"] = count_chars(file_part, ",")
    features["qty_plus_file"] = count_chars(file_part, "+")
    features["qty_asterisk_file"] = count_chars(file_part, "*")
    features["qty_hashtag_file"] = count_chars(file_part, "#")
    features["qty_dollar_file"] = count_chars(file_part, "$")
    features["qty_percent_file"] = count_chars(file_part, "%")
    features["file_length"] = len(file_part)

    # params counts
    features["qty_dot_params"] = count_chars(query, ".")
    features["qty_hyphen_params"] = count_chars(query, "-")
    features["qty_underline_params"] = count_chars(query, "_")
    features["qty_slash_params"] = count_chars(query, "/")
    features["qty_questionmark_params"] = count_chars(query, "?")
    features["qty_equal_params"] = count_chars(query, "=")
    features["qty_at_params"] = count_chars(query, "@")
    features["qty_and_params"] = count_chars(query, "&")
    features["qty_exclamation_params"] = count_chars(query, "!")
    features["qty_space_params"] = count_chars(query, " ")
    features["qty_tilde_params"] = count_chars(query, "~")
    features["qty_comma_params"] = count_chars(query, ",")
    features["qty_plus_params"] = count_chars(query, "+")
    features["qty_asterisk_params"] = count_chars(query, "*")
    features["qty_hashtag_params"] = count_chars(query, "#")
    features["qty_dollar_params"] = count_chars(query, "$")
    features["qty_percent_params"] = count_chars(query, "%")
    features["params_length"] = len(query)
    # tld present in params (naive check)
    features["tld_present_params"] = 1 if re.search(r"\.(" + r"|".join([
        "com","org","net","in","co","edu","gov","info","biz","io","me","app"
    ]) + r")", query) else 0
    # number of params
    try:
        features["qty_params"] = len(parse_qs(query))
    except Exception:
        features["qty_params"] = 0

    # email in URL
    features["email_in_url"] = 1 if re.search(r"[A-Za-z0-9\._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", (parsed.netloc + parsed.path + query)) else 0

    # domain_length and domain_in_ip moved to DOM group (compute here)
    domain = extract_domain(url)
    features["domain_length"] = len(domain)
    features["domain_in_ip"] = 1 if is_ip_string(domain) else 0

    return features

# -------------------------
# NETWORK features (WHOIS, DNS, SSL, redirects, etc.)
# -------------------------
def extract_network_features(url, timeout=4.0):
    full_url = url if url.startswith(("http://", "https://")) else "http://" + url
    domain = extract_domain(full_url)

    features = {}
    # default initial values
    features["time_response"] = -1.0
    features["domain_spf"] = 0
    features["asn_ip"] = -1
    features["time_domain_activation"] = -1
    features["time_domain_expiration"] = -1
    features["qty_ip_resolved"] = 0
    features["qty_nameservers"] = 0
    features["qty_mx_servers"] = 0
    features["ttl_hostname"] = -1
    features["tls_ssl_certificate"] = 0
    features["qty_redirects"] = 0
    features["url_google_index"] = 0
    features["domain_google_index"] = 0
    features["url_shortened"] = 1 if any(s in domain for s in SHORTENERS) else 0

    # Response and redirects via requests
    try:
        r = requests.get(full_url, timeout=timeout, allow_redirects=True)
        features["time_response"] = getattr(r.elapsed, "total_seconds", lambda: None)() if r.elapsed is not None else -1.0
        features["qty_redirects"] = len(r.history) if hasattr(r, "history") else 0
    except Exception:
        # keep defaults
        pass

    # WHOIS: domain dates
    try:
        import whois as _whois
        w = _whois.whois(domain)
        creation = safe_get(w, "creation_date", None)
        expiration = safe_get(w, "expiration_date", None)

        # normalize lists
        if isinstance(creation, list) and creation:
            creation = creation[0]
        if isinstance(expiration, list) and expiration:
            expiration = expiration[0]

        if isinstance(creation, datetime):
            features["time_domain_activation"] = (datetime.now() - creation).days
        if isinstance(expiration, datetime):
            features["time_domain_expiration"] = (expiration - datetime.now()).days
    except Exception:
        # keep defaults
        pass

    # DNS queries
    try:
        ans_a = dns.resolver.resolve(domain, "A")
        features["qty_ip_resolved"] = len(ans_a)
        # attempt to get TTL from rrset
        try:
            features["ttl_hostname"] = ans_a.rrset.ttl
        except Exception:
            features["ttl_hostname"] = -1
    except Exception:
        features["qty_ip_resolved"] = 0
        features["ttl_hostname"] = -1

    try:
        ans_ns = dns.resolver.resolve(domain, "NS")
        features["qty_nameservers"] = len(ans_ns)
    except Exception:
        features["qty_nameservers"] = 0

    try:
        ans_mx = dns.resolver.resolve(domain, "MX")
        features["qty_mx_servers"] = len(ans_mx)
    except Exception:
        features["qty_mx_servers"] = 0

    # domain_spf: check for SPF-like TXT records containing "v=spf1"
    try:
        txts = dns.resolver.resolve(domain, "TXT")
        txts_joined = " ".join([t.to_text().strip('"') for t in txts])
        features["domain_spf"] = 1 if "v=spf1" in txts_joined.lower() else 0
    except Exception:
        features["domain_spf"] = 0

    # SSL certificate presence (try connecting to port 443)
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
            s.settimeout(3)
            s.connect((domain, 443))
            cert = s.getpeercert()
            # if certificate returned, mark 1
            features["tls_ssl_certificate"] = 1
    except Exception:
        features["tls_ssl_certificate"] = 0

    # ASN lookup: try ipwhois if installed, otherwise fallback to simple IP -> -1 or IP integer
    try:
        # get IP from DNS A
        ip_addr = None
        try:
            answers = dns.resolver.resolve(domain, "A")
            if answers:
                ip_addr = answers[0].to_text()
        except Exception:
            try:
                ip_addr = socket.gethostbyname(domain)
            except Exception:
                ip_addr = None

        if _HAS_IPWHOIS and ip_addr:
            try:
                obj = IPWhois(ip_addr)
                res = obj.lookup_rdap(asn_methods=["whois"])
                asn = safe_get(res, "asn", None)
                features["asn_ip"] = int(asn) if asn and str(asn).isdigit() else -1
            except Exception:
                features["asn_ip"] = -1
        else:
            # fallback: store integer conversion of last octet or -1
            if ip_addr:
                try:
                    parts = ip_addr.split(".")
                    features["asn_ip"] = int(parts[-1])
                except Exception:
                    features["asn_ip"] = -1
            else:
                features["asn_ip"] = -1
    except Exception:
        features["asn_ip"] = -1

    return features

# -------------------------
# MASTER: combine all features into a single-row dataframe
# -------------------------
def extract_all_features(url):
    """
    Returns a pandas DataFrame with exactly the feature names used by the model.
    NOTE: Order of columns is not guaranteed here — calling code should reindex
    using feature_groups loaded from disk (feature_groups.pkl) to align order.
    """
    # gather parts
    u = extract_url_features(url)
    d = extract_dom_features(url)
    n = extract_network_features(url)

    # Merge dictionaries; network WHOIS computed time_domain_* should be used.
    # ensure time_domain values propagate to url & dom dictionaries for consistency
    if n.get("time_domain_activation", None) is not None and n.get("time_domain_activation", -1) != -1:
        u["time_domain_activation"] = n["time_domain_activation"]
    if n.get("time_domain_expiration", None) is not None and n.get("time_domain_expiration", -1) != -1:
        u["time_domain_expiration"] = n["time_domain_expiration"]

    # if dom expects domain_length/domain_in_ip, they are computed in dom extractor already

    all_feat = {}
    all_feat.update(u)
    all_feat.update(d)
    all_feat.update(n)

    # Make sure every feature referenced in your feature_groups exists (create safe defaults)
    required = [
        # url_features (as in feature_groups.pkl)
        'qty_dot_url','qty_hyphen_url','qty_underline_url','qty_slash_url','qty_questionmark_url',
        'qty_equal_url','qty_at_url','qty_and_url','qty_exclamation_url','qty_space_url',
        'qty_tilde_url','qty_comma_url','qty_plus_url','qty_asterisk_url','qty_hashtag_url',
        'qty_dollar_url','qty_percent_url','qty_tld_url','length_url','qty_dot_domain',
        'qty_hyphen_domain','qty_underline_domain','qty_slash_domain','qty_questionmark_domain',
        'qty_equal_domain','qty_at_domain','qty_and_domain','qty_exclamation_domain','qty_space_domain',
        'qty_tilde_domain','qty_comma_domain','qty_plus_domain','qty_asterisk_domain','qty_hashtag_domain',
        'qty_dollar_domain','qty_percent_domain','qty_vowels_domain','server_client_domain',
        'time_domain_activation','time_domain_expiration',
        # dom_features
        'qty_dot_directory','qty_hyphen_directory','qty_underline_directory','qty_slash_directory',
        'qty_questionmark_directory','qty_equal_directory','qty_at_directory','qty_and_directory',
        'qty_exclamation_directory','qty_space_directory','qty_tilde_directory','qty_comma_directory',
        'qty_plus_directory','qty_asterisk_directory','qty_hashtag_directory','qty_dollar_directory',
        'qty_percent_directory','directory_length','qty_dot_file','qty_hyphen_file','qty_underline_file',
        'qty_slash_file','qty_questionmark_file','qty_equal_file','qty_at_file','qty_and_file',
        'qty_exclamation_file','qty_space_file','qty_tilde_file','qty_comma_file','qty_plus_file',
        'qty_asterisk_file','qty_hashtag_file','qty_dollar_file','qty_percent_file','file_length',
        'qty_dot_params','qty_hyphen_params','qty_underline_params','qty_slash_params',
        'qty_questionmark_params','qty_equal_params','qty_at_params','qty_and_params',
        'qty_exclamation_params','qty_space_params','qty_tilde_params','qty_comma_params',
        'qty_plus_params','qty_asterisk_params','qty_hashtag_params','qty_dollar_params',
        'qty_percent_params','params_length','tld_present_params','qty_params','email_in_url',
        'domain_length','domain_in_ip',
        # network_features
        'time_response','domain_spf','asn_ip','time_domain_activation','time_domain_expiration',
        'qty_ip_resolved','qty_nameservers','qty_mx_servers','ttl_hostname','tls_ssl_certificate',
        'qty_redirects','url_google_index','domain_google_index','url_shortened'
    ]

    # fill missing with safe defaults
    for r in required:
        if r not in all_feat:
            # choose sensible default: numeric 0 or -1 for unknowns
            if any(x in r for x in ["time_", "ttl", "asn", "time_domain"]):
                all_feat[r] = -1
            else:
                all_feat[r] = 0

    # Return dataframe with one row
    df = pd.DataFrame([all_feat])

    return df
