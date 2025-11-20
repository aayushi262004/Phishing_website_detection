# feature_extractor.py
# Strict/full feature extractor to match feature_groups.pkl
# Requires: requests, tldextract, python-whois, dnspython, bs4 (BeautifulSoup)
# Optional (better ASN): ipwhois
import re
import socket
import ssl
from datetime import datetime
from urllib.parse import urlparse, parse_qs
import time

import dns.resolver
import pandas as pd
import requests
import tldextract
from bs4 import BeautifulSoup # Required for DOM feature extraction

# --- SMOKE TEST: CONFIRM THIS FILE IS LOADED ---
print("--- FEATURE EXTRACTOR V5 LOADED ---")
# ----------------------------------------------
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
VOWELS = "aeiou"
CONSONANTS = "bcdfghjklmnpqrstvwxyz"

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

    ext = tldextract.extract(full_url)
    features["qty_tld_url"] = 1 if parsed.path.endswith("." + (ext.suffix or "")) else 0
    features["length_url"] = len(full_url)

    # Domain-level char counts (domain string)
    domain_str = ext.fqdn or ""
    features["qty_dot_domain"] = count_chars(domain_str, ".")
    features["qty_hyphen_domain"] = count_chars(domain_str, "-")
    features["qty_underline_domain"] = count_chars(domain_str, "_")
    features["qty_slash_domain"] = count_chars(domain_str, "/")
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
    features["qty_vowels_domain"] = sum(1 for c in domain_str.lower() if c in VOWELS)

    # server_client_domain: heuristic -> 1 if domain looks like server/client keywords present
    features["server_client_domain"] = 1 if any(x in domain_str.lower() for x in ["server", "client", "gateway", "admin", "mail"]) else 0

    # time_domain_activation/time_domain_expiration are computed in network but default here
    features["time_domain_activation"] = -1
    features["time_domain_expiration"] = -1

    return features

# -------------------------
# HTML content features (Requires web scraping)
# -------------------------
def _extract_html_features(response, domain, url):
    features = {}

    # Define ALL expected HTML-derived features and set safe defaults
    html_derived_features = [
        'phish_hints', 'contains_form', 'contains_iframe', 'qty_external_iframe', 'qty_external_scripts',
        'qty_external_images', 'qty_external_css', 'qty_external_favicons', 'in_out_image', 'in_out_css',
        'in_out_favicon', 'in_out_form', 'in_out_script', 'qty_divs', 'qty_iframes', 'qty_inputs',
        'qty_scripts', 'qty_labels', 'qty_embeds', 'qty_objects', 'qty_external_links', 'qty_internal_links',
        'qty_hyperlinks', 'ratio_intMedia', 'ratio_extMedia', 'ratio_intHyperlinks', 'ratio_extHyperlinks',
        'ratio_intErrors', 'ratio_extErrors', 'ratio_error', 'login_present', 'external_favicon',
        'url_ip_present', 'tld_present_in_title', 'script_in_body', 'qty_char_set', 'vowel_consonant_ratio',
        'domain_in_body', 'url_in_body', 'domain_in_title', 'url_in_title', 'content_length'
    ]

    # Initialize all 41 features to defaults (0 for counts/ratios, -1 for length)
    for key in html_derived_features:
        if 'ratio' in key or 'length' == key or key in ['content_length']:
            features[key] = -1.0 if 'length' in key else 0.0
        else:
            features[key] = 0

    if response is None or response.status_code != 200:
        return features # Return initialized defaults if request failed

    try:
        soup = BeautifulSoup(response.content, 'html.parser')
    except Exception:
        return features # Return initialized defaults if parsing fails

    # Calculate actual features

    # Basic Counts
    features['qty_divs'] = len(soup.find_all('div'))
    features['qty_iframes'] = len(soup.find_all('iframe'))
    features['qty_inputs'] = len(soup.find_all('input'))
    features['qty_scripts'] = len(soup.find_all('script'))
    features['qty_labels'] = len(soup.find_all('label'))
    features['qty_embeds'] = len(soup.find_all('embed'))
    features['qty_objects'] = len(soup.find_all('object'))

    # Content Length
    try:
        features['content_length'] = len(response.text)
    except:
        features['content_length'] = -1

    # Phishing Hints (Naive check)
    phish_keywords = ["login", "signin", "account", "verify", "update", "password", "bank"]
    body_text = soup.body.get_text() if soup.body else ""
    features['phish_hints'] = 1 if any(k in body_text.lower() for k in phish_keywords) else 0

    # Forms and login fields
    features['contains_form'] = 1 if soup.find('form') else 0
    features['login_present'] = 1 if soup.find('input', {'type': 'password'}) else 0

    # Character set (crude check)
    features['qty_char_set'] = len(response.encoding or 'utf-8')

    # Textual analysis
    vowels_count = sum(1 for c in body_text.lower() if c in VOWELS)
    consonants_count = sum(1 for c in body_text.lower() if c in CONSONANTS)
    features['vowel_consonant_ratio'] = vowels_count / (consonants_count + 1e-6)

    # Internal/External Link Analysis
    all_links = soup.find_all(['a', 'link', 'script', 'img', 'iframe'])

    counts = {
        'external_links': 0, 'internal_links': 0, 'hyperlinks': 0,
        'external_iframe': 0, 'external_scripts_calc': 0, 'external_images_calc': 0,
        'external_css_calc': 0, 'external_favicons_calc': 0,
        'internal_media': 0, 'external_media': 0, 'internal_css': 0,
        'internal_favicon': 0, 'internal_form': 0, 'external_form': 0,
        'internal_script': 0,
    }

    # Helper function to check if a URL is external
    def is_external(href, current_domain):
        if not href or href.startswith(('#', 'mailto:', 'tel:')): return False
        try:
            link_domain = extract_domain(href)
            return link_domain and link_domain != current_domain
        except:
            return False

    for tag in all_links:
        src = tag.get('src') or tag.get('href')
        if not src: continue

        is_ext = is_external(src, domain)

        # Hyperlinks (A tags)
        if tag.name == 'a':
            counts['hyperlinks'] += 1
            if is_ext:
                counts['external_links'] += 1
            else:
                counts['internal_links'] += 1

        # Scripts
        elif tag.name == 'script':
            if is_ext:
                counts['external_scripts_calc'] += 1

        # Images/Media
        elif tag.name == 'img' or tag.name == 'embed' or tag.name == 'object':
            if is_ext:
                counts['external_media'] += 1
            else:
                counts['internal_media'] += 1

        # CSS (Link tags with rel=stylesheet)
        elif tag.name == 'link' and tag.get('rel') == ['stylesheet']:
            if is_ext:
                counts['external_css_calc'] += 1
            else:
                counts['internal_css'] += 1

        # Favicon
        elif tag.name == 'link' and 'icon' in (tag.get('rel') or []):
            if is_ext:
                counts['external_favicons_calc'] += 1
            else:
                counts['internal_favicon'] += 1

        # iFrames
        elif tag.name == 'iframe':
             if is_ext:
                counts['external_iframe'] += 1

    # Map calculated counts to feature names
    features['qty_external_iframe'] = counts['external_iframe']
    features['qty_external_scripts'] = counts['external_scripts_calc']
    features['qty_external_images'] = counts['external_media']
    features['qty_external_css'] = counts['external_css_calc']
    features['qty_external_favicons'] = counts['external_favicons_calc']
    features['qty_external_links'] = counts['external_links']
    features['qty_internal_links'] = counts['internal_links']
    features['qty_hyperlinks'] = counts['hyperlinks']

    # Ratios
    total_media = counts['internal_media'] + counts['external_media']
    total_hyperlinks = counts['external_links'] + counts['internal_links']

    features['ratio_intMedia'] = counts['internal_media'] / (total_media + 1e-6)
    features['ratio_extMedia'] = counts['external_media'] / (total_media + 1e-6)
    features['ratio_intHyperlinks'] = counts['internal_links'] / (total_hyperlinks + 1e-6)
    features['ratio_extHyperlinks'] = counts['external_links'] / (total_hyperlinks + 1e-6)

    # In/Out Ratios (Heuristics based on total internal/external resources/forms/scripts)
    features['in_out_image'] = 1 if counts['external_media'] > counts['internal_media'] else 0
    features['in_out_css'] = 1 if counts['external_css_calc'] > counts['internal_css'] else 0
    features['in_out_favicon'] = 1 if counts['external_favicons_calc'] > counts['internal_favicon'] else 0

    # Forms and Scripts are external if links are external
    form_tag = soup.find('form')
    features['in_out_form'] = 1 if form_tag and is_external(form_tag.get('action'), domain) else 0
    script_tag = soup.find('script')
    features['in_out_script'] = 1 if script_tag and is_external(script_tag.get('src'), domain) else 0

    # Simple checks
    features['external_favicon'] = features['qty_external_favicons'] > 0
    features['script_in_body'] = 1 if soup.body and soup.body.find('script') else 0

    # URL/Domain presence in title/body
    ext = tldextract.extract(url)
    title_text = soup.title.string.lower() if soup.title and soup.title.string else ""
    body_plain_text = soup.body.get_text().lower() if soup.body else ""

    features['domain_in_body'] = 1 if domain.lower() in body_plain_text else 0
    features['url_in_body'] = 1 if url.lower() in body_plain_text else 0
    features['domain_in_title'] = 1 if domain.lower() in title_text else 0
    features['url_in_title'] = 1 if url.lower() in title_text else 0
    features['tld_present_in_title'] = 1 if ext.suffix in title_text else 0

    # Placeholder features (Errors and Google Index are hard to implement reliably without external services/access)
    features['url_ip_present'] = features.get('domain_in_ip', 0) # Use the value calculated in URL-only

    # Note: ratio_intErrors, ratio_extErrors, ratio_error remain 0.0 defaults

    return features


# -------------------------
# DOM / path / file / params features (URL-derived only)
# -------------------------
def extract_dom_features_url_only(url):
    parsed = urlparse(url if url.startswith(("http://", "https://")) else "http://" + url)
    query = parsed.query or ""

    features = {}

    # params counts (17 character counts + qty_params)
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

    features["qty_params"] = len(parse_qs(query))

    # domain_length and domain_in_ip
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
    # default initial values (must match the 14 features exactly)
    features["time_response"] = -1.0
    features["domain_spf"] = 0
    features["asn_ip"] = -1
    features["time_domain_activation"] = -1
    features["time_domain_expiration"] = -1
    features["qty_ip_resolved"] = 0
    features["qty_nameservers"] = 0
    features["qty_mx_servers"] = 0
    features["ttl_a_record"] = -1 # Matches name in feature list (was ttl_hostname in calc)
    features["url_accessible"] = 0 # Matches name in feature list (was tls_ssl_certificate in calc)
    features["qty_redirects"] = 0
    features["url_google_index"] = 0 # Placeholder: Hard to calculate reliably
    features["domain_google_index"] = 0 # Placeholder: Hard to calculate reliably
    features["url_shortened"] = 1 if any(s in domain for s in SHORTENERS) else 0

    # Response and redirects via requests
    r = None
    try:
        start_time = time.time()
        r = requests.get(full_url, timeout=timeout, allow_redirects=True, verify=False)
        features["time_response"] = time.time() - start_time
        features["qty_redirects"] = len(r.history) if hasattr(r, "history") else 0
        features["url_accessible"] = 1 # Mark as accessible if request was successful
    except requests.exceptions.RequestException:
        pass # keep defaults

    # WHOIS: domain dates
    try:
        import whois as _whois
        w = _whois.whois(domain)
        creation = safe_get(w, "creation_date", None)
        expiration = safe_get(w, "expiration_date", None)

        if isinstance(creation, list) and creation: creation = creation[0]
        if isinstance(expiration, list) and expiration: expiration = expiration[0]

        if isinstance(creation, datetime): features["time_domain_activation"] = (datetime.now() - creation).days
        if isinstance(expiration, datetime): features["time_domain_expiration"] = (expiration - datetime.now()).days
    except Exception:
        pass # keep defaults

    # DNS queries
    try:
        ans_a = dns.resolver.resolve(domain, "A")
        features["qty_ip_resolved"] = len(ans_a)
        try:
            # Storing the actual calculated value in a temporary key
            ttl_val = ans_a.rrset.ttl
            features["ttl_a_record"] = ttl_val
        except Exception: features["ttl_a_record"] = -1
    except Exception:
        features["qty_ip_resolved"] = 0
        features["ttl_a_record"] = -1

    try: features["qty_nameservers"] = len(dns.resolver.resolve(domain, "NS"))
    except Exception: features["qty_nameservers"] = 0

    try: features["qty_mx_servers"] = len(dns.resolver.resolve(domain, "MX"))
    except Exception: features["qty_mx_servers"] = 0

    # domain_spf
    try:
        txts = dns.resolver.resolve(domain, "TXT")
        txts_joined = " ".join([t.to_text().strip('"') for t in txts])
        features["domain_spf"] = 1 if "v=spf1" in txts_joined.lower() else 0
    except Exception:
        features["domain_spf"] = 0

    # SSL certificate presence (Optional check, but useful for url_accessible context)
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
            s.settimeout(3)
            s.connect((domain, 443))
            s.getpeercert()
            # If we successfully get cert, that's another indicator of accessibility
    except Exception:
        pass

    # ASN lookup
    try:
        ip_addr = None
        try: ip_addr = dns.resolver.resolve(domain, "A")[0].to_text()
        except Exception: ip_addr = socket.gethostbyname(domain) if not is_ip_string(domain) else domain

        if _HAS_IPWHOIS and ip_addr:
            obj = IPWhois(ip_addr)
            res = obj.lookup_rdap(asn_methods=["whois"], inc_raw=False)
            asn = safe_get(res, "asn", -1)
            features["asn_ip"] = int(asn) if asn and str(asn).isdigit() else -1
        else:
            if ip_addr and not is_ip_string(domain):
                features["asn_ip"] = int(ip_addr.split(".")[-1])
            else:
                features["asn_ip"] = -1
    except Exception:
        features["asn_ip"] = -1

    return features, r

# -------------------------
# MASTER: combine all features into a single-row dataframe
# -------------------------
def extract_all_features(url):
    u = extract_url_features(url)
    d_url = extract_dom_features_url_only(url)
    n, response = extract_network_features(url)

    # Domain features derived from HTML content (41 features)
    d_html = _extract_html_features(response, extract_domain(url), url)

    # Merge dictionaries
    all_feat = {}
    all_feat.update(u)
    all_feat.update(d_url)
    all_feat.update(d_html)
    all_feat.update(n)

    # Propagate WHOIS dates to URL group features
    if 'time_domain_activation' in n and n['time_domain_activation'] != -1:
        all_feat["time_domain_activation"] = n["time_domain_activation"]
    if 'time_domain_expiration' in n and n['time_domain_expiration'] != -1:
        all_feat["time_domain_expiration"] = n["time_domain_expiration"]

    # Return dataframe with one row
    return pd.DataFrame([all_feat])
