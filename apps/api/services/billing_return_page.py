"""Billing return-page rendering helpers."""

from __future__ import annotations

from html import escape
from urllib.parse import quote


def normalize_billing_status(raw_billing: str | None) -> str:
    billing = str(raw_billing or "").strip().lower()
    if billing in {"failed", "fail", "error"}:
        return "cancel"
    if billing == "succeeded":
        return "success"
    return billing


def render_billing_return_html(
    *,
    billing: str,
    language: str,
    app_return_url: str | None,
) -> str:
    is_zh = language == "zh"

    if billing == "success":
        status_kind = "success"
        tone = "#16A34A"
    elif billing in {"cancel"}:
        status_kind = "failed"
        tone = "#DC2626"
    else:
        status_kind = "neutral"
        tone = "#2563EB"

    if is_zh:
        title_map = {
            "success": "支付已完成",
            "failed": "支付未完成",
            "neutral": "账单状态已更新",
        }
        subtitle_map = {
            "success": "你的 Stripe 支付已成功提交。",
            "failed": "本次支付未完成，你可以随时重新尝试。",
            "neutral": "你可以返回应用查看当前订阅状态。",
        }
        thanks_map = {
            "success": "感谢你的支持，我们正在后台同步你的订阅信息。",
            "failed": "感谢你体验 Minsy。支付失败通常与银行卡、网络或 Stripe 校验相关。",
            "neutral": "感谢你使用账单流程，我们正在同步最新账单状态。",
        }
        badge_map = {
            "success": "成功",
            "failed": "失败",
            "neutral": "同步中",
        }
        support_hint = "如果遇到付款问题、功能 bug，或者想优先体验最新功能，欢迎加入我们的 Telegram demo 用户群，直接与我们沟通。"
        group_label = "Telegram 演示用户群"
        join_hint = "扫码或点击下方链接即可加入："
        copy_hint = "若无法打开链接，请手动复制。"
        action_label = "打开订阅管理"
    else:
        title_map = {
            "success": "Payment completed",
            "failed": "Payment not completed",
            "neutral": "Billing status updated",
        }
        subtitle_map = {
            "success": "Your Stripe payment went through successfully.",
            "failed": "Your payment was not completed. You can retry anytime.",
            "neutral": "You can return to the app and review your subscription status.",
        }
        thanks_map = {
            "success": "Thanks for your support. We are syncing your subscription details in the background.",
            "failed": "Thanks for trying Minsy. Payment can fail due to card, network, or Stripe validation issues.",
            "neutral": "Thanks for visiting our billing flow. Your latest billing state is being synchronized.",
        }
        badge_map = {
            "success": "Success",
            "failed": "Failed",
            "neutral": "In sync",
        }
        support_hint = "If you hit payment issues, feature bugs, or want early access to the newest beta capabilities, join our Telegram demo group and talk to us directly."
        group_label = "Telegram demo user group"
        join_hint = "Scan the QR code or open the link below:"
        copy_hint = "If opening fails, copy the link manually."
        action_label = "Open subscription"

    title = title_map.get(status_kind, title_map["neutral"])
    subtitle = subtitle_map.get(status_kind, subtitle_map["neutral"])
    thanks = thanks_map.get(status_kind, thanks_map["neutral"])
    badge = badge_map.get(status_kind, badge_map["neutral"])

    action_html = ""
    if app_return_url:
        action_html = (
            "<div class=\"actions\">"
            f"<a class=\"button primary\" href=\"{escape(app_return_url)}\" rel=\"noreferrer\">"
            f"{escape(action_label)}"
            "</a>"
            "</div>"
        )

    ring_html = ""
    if status_kind in {"success", "failed"}:
        ring_html = f"<div class=\"ring-orbit ring-{status_kind}\" aria-hidden=\"true\"></div>"

    telegram_link = "https://t.me/+vbxwOIwswK43Yzc1"
    qr_src = (
        "https://api.qrserver.com/v1/create-qr-code/?size=220x220&data="
        + quote(telegram_link, safe="")
    )

    return f"""<!doctype html>
<html lang=\"{escape(language)}\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --bg: #ffffff;
      --surface-subtle: #fafafa;
      --border: #e5e5e5;
      --text-primary: #0d0d0d;
      --text-secondary: #474747;
      --text-muted: #6b6b6b;
      --accent: #3b82f6;
      --card-shadow-1: 0 1px 3px rgba(0,0,0,0.06);
      --card-shadow-2: 0 1px 2px rgba(0,0,0,0.04);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: Inter, \"PingFang SC\", \"Hiragino Sans GB\", \"Microsoft YaHei\", \"Noto Sans CJK SC\", \"Source Han Sans SC\", sans-serif;
      color: var(--text-primary);
      background:
        radial-gradient(circle at 15% 10%, rgba(59,130,246,0.08), transparent 34%),
        radial-gradient(circle at 85% 80%, rgba(24,24,27,0.05), transparent 30%),
        var(--bg);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }}
    .card {{
      width: min(760px, 100%);
      border-radius: 12px;
      border: 1px solid var(--border);
      background: var(--bg);
      box-shadow: var(--card-shadow-1), var(--card-shadow-2);
      padding: 22px 24px 20px;
      position: relative;
      z-index: 1;
    }}
    .card-shell {{
      width: min(760px, 100%);
      position: relative;
      border-radius: 12px;
    }}
    .ring-orbit {{
      position: absolute;
      inset: -3px;
      border-radius: 14px;
      pointer-events: none;
      overflow: hidden;
      z-index: 0;
      filter: saturate(1.08);
    }}
    .ring-orbit::before {{
      content: \"\";
      position: absolute;
      left: -50%;
      top: -50%;
      width: 200%;
      height: 200%;
      background: conic-gradient(
        from 0deg,
        transparent 0deg,
        transparent 300deg,
        var(--ring-color) 328deg,
        transparent 360deg
      );
      animation: ring-spin 3s linear infinite;
      filter: blur(10px);
      opacity: 0.9;
    }}
    .ring-success {{
      --ring-color: rgba(22, 163, 74, 0.88);
    }}
    .ring-failed {{
      --ring-color: rgba(220, 38, 38, 0.88);
    }}
    @keyframes ring-spin {{
      0% {{
        transform: rotate(0deg);
      }}
      100% {{
        transform: rotate(360deg);
      }}
    }}
    .header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
    }}
    .title {{
      margin: 0;
      font-size: 24px;
      line-height: 1.3;
      font-weight: 600;
      color: var(--text-primary);
    }}
    .subtitle {{
      margin: 6px 0 0;
      font-size: 13px;
      line-height: 1.5;
      color: var(--text-secondary);
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 999px;
      border: 1px solid {tone};
      background: rgba(255, 255, 255, 0.9);
      color: {tone};
      padding: 4px 8px;
      font-size: 11px;
      line-height: 1.15;
      font-weight: 700;
      white-space: nowrap;
      margin-top: 2px;
    }}
    .thanks {{
      margin: 18px 0 0;
      font-size: 13px;
      line-height: 1.55;
      color: var(--text-primary);
    }}
    .support {{
      margin: 8px 0 0;
      font-size: 13px;
      line-height: 1.55;
      color: var(--text-secondary);
    }}
    .telegram {{
      margin-top: 16px;
      width: 100%;
      padding: 12px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--surface-subtle);
      display: grid;
      grid-template-columns: 110px 1fr;
      gap: 12px;
      align-items: start;
    }}
    .qr-wrap {{
      width: 110px;
      height: 110px;
      border-radius: 8px;
      border: 1px solid var(--border);
      overflow: hidden;
      background: var(--surface-subtle);
    }}
    .qr-wrap img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}
    .group-label {{
      margin: 6px 0 0;
      font-size: 12px;
      line-height: 1.25;
      font-weight: 600;
      color: var(--text-secondary);
    }}
    .join-hint {{
      margin: 10px 0 0;
      font-size: 12px;
      line-height: 1.35;
      color: var(--text-primary);
    }}
    .tg-link {{
      margin-top: 10px;
      display: inline-block;
      font-size: 12px;
      line-height: 1.35;
      font-weight: 600;
      color: var(--accent);
      text-decoration: underline;
      text-decoration-color: var(--accent);
      word-break: break-all;
    }}
    .copy-hint {{
      margin: 6px 0 0;
      font-size: 11px;
      line-height: 1.35;
      color: var(--text-secondary);
    }}
    .actions {{
      margin-top: 16px;
      display: flex;
      gap: 10px;
    }}
    .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 34px;
      border-radius: 8px;
      padding: 8px 16px;
      text-decoration: none;
      font-size: 12px;
      line-height: 1.1;
      font-weight: 600;
      border: 1px solid var(--border);
      color: var(--text-primary);
      background: #fff;
    }}
    .button.primary {{
      background: #18181b;
      border-color: #18181b;
      color: #fafafa;
    }}
    @media (max-width: 560px) {{
      .header {{
        flex-direction: column;
        align-items: flex-start;
      }}
      .telegram {{
        grid-template-columns: 1fr;
      }}
      .group-label {{
        margin-top: 0;
      }}
    }}
  </style>
</head>
<body>
  <div class=\"card-shell status-{escape(status_kind)}\">
    {ring_html}
    <main class=\"card\">
      <div class=\"header\">
        <div>
          <h1 class=\"title\">{escape(title)}</h1>
          <p class=\"subtitle\">{escape(subtitle)}</p>
        </div>
        <span class=\"badge\">{escape(badge)}</span>
      </div>
      <p class=\"thanks\">{escape(thanks)}</p>
      <p class=\"support\">{escape(support_hint)}</p>
      <section class=\"telegram\">
        <div class=\"qr-wrap\">
          <img src=\"{escape(qr_src)}\" alt=\"Telegram QR code\">
        </div>
        <div>
          <p class=\"group-label\">{escape(group_label)}</p>
          <p class=\"join-hint\">{escape(join_hint)}</p>
          <a class=\"tg-link\" href=\"{escape(telegram_link)}\" rel=\"noreferrer noopener\">{escape(telegram_link)}</a>
          <p class=\"copy-hint\">{escape(copy_hint)}</p>
        </div>
      </section>
      {action_html}
    </main>
  </div>
</body>
</html>"""
