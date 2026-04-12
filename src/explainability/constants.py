"""Shared names used by explainability reports."""

MODALITIES = ["url", "text", "visual", "html"]

URL_FEATURE_NAMES = [
    "url_length",
    "url_num_dots",
    "url_num_hyphens",
    "url_num_slashes",
    "url_num_digits",
    "url_num_special",
    "url_has_ip",
    "url_has_https",
    "url_subdomain_count",
]

HTML_FEATURE_NAMES = [
    "html_form_count",
    "html_input_count",
    "html_has_password",
    "html_script_count",
    "html_iframe_count",
    "html_meta_refresh",
    "html_external_link_ratio",
    "html_visible_text_length",
]

DASHBOARD_EXCLUDE_COLUMNS = {
    "sample_id",
    "feature_index",
    "split",
    "url",
    "label",
    "predicted_label",
    "fusion_phishing_prob",
}
