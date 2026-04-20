# ===========================================================
# default_responses.py — Fallback templates per tag
# ===========================================================
"""
Handcrafted generic responses used when the QA matcher
has no close match. These are NOT pulled from Excel rows —
Excel provides specific Q&A pairs, these provide fallbacks.

To add a new tag: add an entry here with en, ja, type, and labels.
If a tag exists in Excel but not here, export_responses() will
fall back to the first Excel row for that tag.
"""

DEFAULT_RESPONSES = {
    "password": {
        "en": (
            "I can help with your account access.\n\n"
            "Quick options:\n"
            "• Reset password → Settings > Account > Reset Password\n"
            "• Forgot password → Tap \"Forgot Password\" on login screen\n"
            "• Locked out → Wait 15 min or reset via email\n\n"
            "Reset links expire in 30 minutes. Need more help? Just say \"talk to support\"."
        ),
        "ja": (
            "アカウントへのアクセスについてお手伝いします。\n\n"
            "方法：\n"
            "• パスワードリセット → 設定 > アカウント > パスワードリセット\n"
            "• パスワードを忘れた → ログイン画面の「パスワードを忘れた」をタップ\n"
            "• ロックアウト → 15分待つか、メールでリセット\n\n"
            "リセットリンクは30分有効です。さらにサポートが必要な場合は「サポートに連絡」とお伝えください。"
        ),
        "type": "answer",
        "label_en": "Account & Password",
        "label_ja": "アカウント・パスワード",
    },
    "subscription": {
        "en": (
            "Here's how to manage your subscription:\n\n"
            "• Cancel → Settings > Subscription > Cancel Plan\n"
            "• Downgrade → Switch to free plan to keep basic features\n"
            "• Change plan → Settings > Subscription > Change Plan\n\n"
            "Your access continues until the current billing period ends."
        ),
        "ja": (
            "サブスクリプションの管理方法：\n\n"
            "• 解約 → 設定 > サブスクリプション > キャンセル\n"
            "• ダウングレード → 基本機能を維持する無料プランへ変更\n"
            "• プラン変更 → 設定 > サブスクリプション > プラン変更\n\n"
            "現在の請求期間が終了するまでアクセスは継続されます。"
        ),
        "type": "answer",
        "label_en": "Subscription",
        "label_ja": "サブスクリプション",
    },
    "billing": {
        "en": (
            "I can help with billing questions.\n\n"
            "• View history → Settings > Billing\n"
            "• Download receipt → Billing page > Invoice\n"
            "• Double charge → Usually resolves in 3-5 business days\n"
            "• Refund → Available within 14 days of purchase\n\n"
            "For urgent billing issues, say \"talk to support\"."
        ),
        "ja": (
            "請求についてお手伝いします。\n\n"
            "• 履歴確認 → 設定 > 請求\n"
            "• 領収書 → 請求ページ > インボイス\n"
            "• 二重請求 → 通常3〜5営業日で解決\n"
            "• 返金 → 購入から14日以内に対応可能\n\n"
            "緊急の場合は「サポートに連絡」とお伝えください。"
        ),
        "type": "answer",
        "label_en": "Billing & Payments",
        "label_ja": "請求・お支払い",
    },
    "howto": {
        "en": (
            "Here are some popular features:\n\n"
            "• Dark Mode → Settings > Display > Dark Mode\n"
            "• Notifications → Settings > Notifications\n"
            "• Export Data → Settings > Privacy > Export Data\n"
            "• Language → Settings > General > Language\n"
            "• Multi-device → Sign in with same account\n\n"
            "What specific feature would you like help with?"
        ),
        "ja": (
            "よく使われる機能：\n\n"
            "• ダークモード → 設定 > 表示 > ダークモード\n"
            "• 通知設定 → 設定 > 通知\n"
            "• データエクスポート → 設定 > プライバシー > エクスポート\n"
            "• 言語変更 → 設定 > 一般 > 言語\n"
            "• マルチデバイス → 同じアカウントでログイン\n\n"
            "特定の機能について知りたいことはありますか？"
        ),
        "type": "answer",
        "label_en": "How to use features",
        "label_ja": "機能の使い方",
    },
    "bug": {
        "en": (
            "Sorry about that! Let's get this fixed.\n\n"
            "Quick fixes to try first:\n"
            "• Force close and reopen the app\n"
            "• Check for app updates\n"
            "• Restart your device\n\n"
            "Still broken? Submit a report:\n"
            "→ Settings > Help > Report a Bug\n"
            "(Include device model + app version)"
        ),
        "ja": (
            "ご不便をおかけして申し訳ありません。\n\n"
            "まずお試しください：\n"
            "• アプリを強制終了して再起動\n"
            "• アプリのアップデートを確認\n"
            "• デバイスを再起動\n\n"
            "解決しない場合はレポートを送信：\n"
            "→ 設定 > ヘルプ > バグを報告\n"
            "（デバイスモデルとアプリバージョンをご記入ください）"
        ),
        "type": "support",
        "label_en": "Report a bug",
        "label_ja": "バグ報告",
    },
    "support": {
        "en": (
            "I'll connect you with our team.\n\n"
            "→ Settings > Help > Contact Support\n\n"
            "For faster response, include:\n"
            "• Your account email\n"
            "• Brief description of the issue\n"
            "• Screenshots if applicable\n\n"
            "Response time: within 24 hours (Mon-Fri 9AM-6PM JST)."
        ),
        "ja": (
            "サポートチームにおつなぎします。\n\n"
            "→ 設定 > ヘルプ > サポートに連絡\n\n"
            "迅速な対応のため：\n"
            "• アカウントのメールアドレス\n"
            "• 問題の簡単な説明\n"
            "• スクリーンショット（あれば）\n\n"
            "対応時間：24時間以内（月〜金 9:00-18:00 JST）"
        ),
        "type": "support",
        "label_en": "Contact support team",
        "label_ja": "サポートチームに連絡",
    },
    "unknown": {
        "en": (
            "That doesn't seem related to our app, but I can help with:\n\n"
            "• Account & Password\n"
            "• Subscription management\n"
            "• Billing questions\n"
            "• App features & how-to\n"
            "• Bug reports\n\n"
            "Try asking something like \"reset my password\" or \"cancel my plan\"."
        ),
        "ja": (
            "アプリに関するご質問ではないようですが、以下のサポートが可能です：\n\n"
            "• アカウント・パスワード\n"
            "• サブスクリプション管理\n"
            "• 請求に関する質問\n"
            "• 機能の使い方\n"
            "• バグ報告\n\n"
            "「パスワードを変更したい」や「プランを解約したい」などとお聞きください。"
        ),
        "type": "reject",
        "label_en": "Other",
        "label_ja": "その他",
    },
    "greeting": {
        "en": (
            "Hi! I'm your support assistant.\n\n"
            "I can help with:\n"
            "• Account & Password\n"
            "• Subscriptions & Billing\n"
            "• App features\n"
            "• Bug reports\n\n"
            "Just type your question. You can switch to Japanese anytime."
        ),
        "ja": (
            "こんにちは！サポートアシスタントです。\n\n"
            "お手伝いできること：\n"
            "• アカウント・パスワード\n"
            "• サブスクリプション・請求\n"
            "• アプリの機能\n"
            "• バグ報告\n\n"
            "ご質問をどうぞ。英語への切り替えもいつでも可能です。"
        ),
        "type": "answer",
        "label_en": "General help",
        "label_ja": "一般的なヘルプ",
    },
}
