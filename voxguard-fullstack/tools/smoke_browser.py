"""Optional browser integration check against a running local service.

Install playwright and its Chromium browser separately. No model accuracy claim.
"""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=ROOT / "test-results")
    parser.add_argument("--model-id", default="example-localizer")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1050}, accept_downloads=True)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.base_url)
        expect(page.locator("#model-select")).to_be_enabled(timeout=30000)
        expect(page.locator("#score-value")).to_have_text("—")
        expect(page.locator("#analyze-button")).to_be_disabled()
        page.select_option("#model-select", args.model_id)
        page.set_input_files("#file-input", str(ROOT / "samples/audio/CON_D_0000000.wav"))
        expect(page.locator("#analyze-button")).to_be_enabled()
        page.click("#analyze-button")
        expect(page.locator("#export-report")).to_be_enabled(timeout=120000)
        expect(page.locator("#analyze-button")).to_be_enabled(timeout=30000)
        expect(page.locator("#play-button")).to_be_enabled()
        with page.expect_download() as download:
            page.click("#export-report")
        report_path = args.output / f"{args.model_id}-report.json"
        download.value.save_as(report_path)
        report = json.loads(report_path.read_text())
        record = report["analysis"]
        assert record["status"] == "succeeded"
        assert record["model"]["id"] == args.model_id
        assert record["is_mock"] is True  # These smoke-check model IDs are test adapters.
        assert "%" not in page.locator("#score-value").inner_text()
        assert "模拟" in page.locator("#result-demo-label").inner_text()
        segments = record["result"]["localization"]["segments"]
        assert page.locator(".segment-item").count() == len(segments)
        assert page.locator("#audio-player").evaluate("a => a.src.startsWith('blob:')")
        if segments:
            page.locator(".segment-item").first.click()
            position = page.locator("#audio-player").evaluate("a => { const t = a.currentTime; a.pause(); return t; }")
            assert abs(position - segments[0]["start_sec"]) < .5
        page.evaluate("window.scrollTo(0, 0)")
        page.screenshot(path=str(args.output / f"{args.model_id}-desktop.png"), full_page=True, animations="disabled")
        page.select_option("#analysis-mode", "detect")
        page.click("#analyze-button")
        expect(page.locator("#export-report")).to_be_enabled(timeout=120000)
        expect(page.locator("#analyze-button")).to_be_enabled(timeout=30000)
        expect(page.locator("#segment-list")).to_contain_text("未执行时间片段定位")
        if args.model_id == "example-localizer":
            page.select_option("#model-select", "example-detector")
            expect(page.locator("#analysis-mode")).to_have_value("detect")
            page.click("#analyze-button")
            expect(page.locator("#export-report")).to_be_enabled(timeout=120000)
            expect(page.locator("#analyze-button")).to_be_enabled(timeout=30000)
            expect(page.locator("#segment-list")).to_contain_text("不支持时间片段定位")
        page.set_viewport_size({"width": 390, "height": 844})
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_function("document.querySelector('#sidebar').getBoundingClientRect().right <= 1")
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 2")
        page.click("#menu-button")
        expect(page.locator("#menu-button")).to_have_attribute("aria-expanded", "true")
        page.keyboard.press("Escape")
        expect(page.locator("#menu-button")).to_have_attribute("aria-expanded", "false")
        page.wait_for_function("document.querySelector('#sidebar').getBoundingClientRect().right <= 1")
        page.screenshot(path=str(args.output / f"{args.model_id}-mobile.png"), full_page=True, animations="disabled")
        page.goto(args.base_url + "/#history")
        expect(page.locator("#history-body tr")).to_have_count(3 if args.model_id == "example-localizer" else 2)
        assert not errors, errors
        print(json.dumps({"model": args.model_id, "browser_errors": errors, "is_mock": record["is_mock"], "checks": ["initial state", "HTTP upload and polling", "score semantics", "localization", "WAV playback", "JSON export", "detection-only state", "mobile overflow", "history persistence"]}, ensure_ascii=False, indent=2))
        context.close()
        browser.close()


if __name__ == "__main__":
    main()
