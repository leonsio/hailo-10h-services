"use strict";
/** Shared browser language preference and translations for all service pages. */
window.HailoUI = (() => {
  let config = {}, language = "de", choice = "auto", data = {}, generation = 0;
  const listeners = new Set();
  const savedChoice = () => { try { return localStorage.getItem("hailo-ui-language") || "auto"; } catch { return choice; } };
  function chooseLanguage(value, settings = config) {
    const supported = settings?.ui_languages || ["de", "en", "es", "fr", "it", "nl", "pt", "ru"];
    if (supported.includes(value)) return value;
    for (const locale of navigator.languages || [navigator.language || ""]) {
      const code = locale.toLowerCase().split("-")[0];
      if (supported.includes(code)) return code;
    }
    return supported.includes(settings?.service_language) ? settings.service_language : "de";
  }
  const t = (key, values = {}) => (data.ui?.[key] || key).replace(/\{(\w+)\}/g, (_, name) => String(values[name] ?? ""));
  function apply() {
    document.documentElement.lang = language;
    for (const node of document.querySelectorAll("[data-i18n]")) node.textContent = t(node.dataset.i18n);
    for (const [attribute, property] of [["placeholder", "placeholder"], ["alt", "alt"], ["aria", "aria-label"]]) {
      for (const node of document.querySelectorAll(`[data-i18n-${attribute}]`)) node.setAttribute(property, t(node.getAttribute(`data-i18n-${attribute}`)));
    }
    const selector = document.getElementById("ui-language");
    if (selector) {
      selector.replaceChildren(new Option(t("ui_81"), "auto"), ...(config.ui_languages || ["de", "en", "es", "fr", "it", "nl", "pt", "ru"]).map(code => new Option(data.language_names?.[code] || code, code)));
      selector.value = choice;
    }
  }
  async function load(value = savedChoice(), settings = config) {
    config = settings || {};
    const next = chooseLanguage(value), id = ++generation;
    const response = await fetch(`/ui/locales/${next}.json`, {cache: "no-cache"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const catalogue = await response.json();
    if (id !== generation) return;
    data = catalogue; language = next; choice = value;
    apply();
    for (const listener of listeners) await listener();
  }
  async function setChoice(value) {
    await load(value);
    if (choice !== value) return;
    try { localStorage.setItem("hailo-ui-language", value); } catch { /* Works for this page even if storage is disabled. */ }
  }
  window.addEventListener("storage", event => {
    if (event.key === "hailo-ui-language") void load(event.newValue || "auto").catch(() => {});
  });
  return {savedChoice, chooseLanguage, t, apply, load, setChoice,
    subscribe: callback => listeners.add(callback),
    get language() { return language; }, get choice() { return choice; }, get catalogue() { return data; }};
})();
