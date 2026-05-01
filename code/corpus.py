"""
Support Corpus - Grounded knowledge base for HackerRank, Claude, and Visa.
All information sourced directly from official support pages.
"""

SUPPORT_CORPUS = {
    "hackerrank": {
        "domain_description": "HackerRank is a technical hiring platform with Screen (assessments), Interviews, Engage (candidate experience), Chakra (AI), SkillUp (learning), and Library (question library).",
        "categories": [
            "Screen / Assessments",
            "Interviews",
            "Engage",
            "SkillUp",
            "Library / Questions",
            "Account / Settings",
            "Integrations",
            "Billing",
            "General Help",
            "Security",
            "Chakra / AI Features",
        ],
        "escalate_triggers": [
            "billing dispute",
            "account compromise",
            "cheating investigation",
            "legal",
            "data breach",
            "GDPR",
            "CCPA",
            "API credentials",
            "SSO configuration",
            "bulk user import",
            "custom SLA",
            "enterprise contract",
            "impersonation",
            "fraud",
        ],
        "faq": [
            {
                "q": "How do I reset my password?",
                "a": "Go to the HackerRank login page and click 'Forgot Password'. Enter your registered email address and you will receive a password reset link. The link expires after 24 hours. If you do not receive the email, check your spam folder or contact support.",
                "source": "https://support.hackerrank.com/articles/7046498277-update-or-reset-password"
            },
            {
                "q": "What URLs and IPs should I allowlist/safelist for HackerRank?",
                "a": "You should allowlist HackerRank's domains and IP ranges for proper platform access. Please refer to the official Safelist/Allowlist article at https://support.hackerrank.com/articles/6769658535-safelist-or-allowlist-urls-for-hackerrank for the current list. This is important for corporate firewalls and proctoring environments.",
                "source": "https://support.hackerrank.com/articles/6769658535-safelist-or-allowlist-urls-for-hackerrank"
            },
            {
                "q": "What is the execution environment for code challenges?",
                "a": "HackerRank uses sandboxed execution environments for code submissions. The exact runtime versions and supported languages depend on the test configuration set by the employer or HackerRank. You can view supported languages and environments in the assessment settings or the Execution Environment article.",
                "source": "https://support.hackerrank.com/articles/6693750503-execution-environment"
            },
            {
                "q": "When is HackerRank's maintenance window?",
                "a": "HackerRank has scheduled maintenance windows. For current maintenance schedules and notifications, refer to the Maintenance Window Notification article on the support site or check the HackerRank status page.",
                "source": "https://support.hackerrank.com/articles/2086891729-hackerrank-maintenance-window-notification"
            },
            {
                "q": "What is impersonation detection?",
                "a": "HackerRank uses AI-based impersonation detection to verify candidate identity during assessments. It checks for signs that someone other than the candidate is taking the test. Results are flagged for recruiter review and are not used as definitive proof of cheating.",
                "source": "https://support.hackerrank.com/articles/7825915809-impersonation-detection"
            },
            {
                "q": "How do I create a test or assessment in HackerRank Screen?",
                "a": "In HackerRank Screen, go to Tests > Create Test. You can choose from the question library, set a time limit, configure proctoring options, and invite candidates. Detailed steps are in the Screen documentation collection.",
                "source": "https://support.hackerrank.com/collections/1453467047-hackerrank-screen"
            },
            {
                "q": "How do I invite candidates to a HackerRank test?",
                "a": "After creating a test in Screen, you can invite candidates via email directly from the platform, share a public test link, or integrate with your ATS. Go to the test dashboard and click 'Invite Candidates'.",
                "source": "https://support.hackerrank.com/collections/1453467047-hackerrank-screen"
            },
            {
                "q": "What integrations does HackerRank support?",
                "a": "HackerRank supports integrations with major ATS platforms (Greenhouse, Lever, Workday, iCIMS, etc.), SSO providers (Okta, Azure AD, OneLogin), and HRIS tools. There are 94+ integration articles covering setup and troubleshooting.",
                "source": "https://support.hackerrank.com/collections/7654924072-integrations-1"
            },
            {
                "q": "What is HackerRank Engage?",
                "a": "HackerRank Engage is the candidate engagement product that helps companies improve candidate experience, communication, and employer brand throughout the hiring process.",
                "source": "https://support.hackerrank.com/collections/4054400338-engage-"
            },
            {
                "q": "What is Chakra?",
                "a": "Chakra is HackerRank's AI-powered product suite that helps with AI-assisted hiring workflows and intelligent question recommendations.",
                "source": "https://support.hackerrank.com/collections/9492939711-chakra"
            },
            {
                "q": "How do I report a candidate for cheating?",
                "a": "If you suspect cheating, review the assessment report for plagiarism flags, copy-paste events, and proctoring data (if enabled). You can flag the submission in the platform. For serious incidents, escalate to your HackerRank account manager or submit a support ticket.",
                "source": "https://support.hackerrank.com/collections/1453467047-hackerrank-screen"
            },
            {
                "q": "What is HackerRank SkillUp?",
                "a": "HackerRank SkillUp is the learning and upskilling product, offering coding practice, learning paths, and skill assessments for developers.",
                "source": "https://support.hackerrank.com/collections/6175643472-skillup"
            },
            {
                "q": "How do I access the HackerRank question library?",
                "a": "The Library in HackerRank contains 51+ articles on question management. Admins and recruiters can browse, filter, and add questions to their tests from the Library tab in Screen.",
                "source": "https://support.hackerrank.com/collections/9271153455-library"
            },
            {
                "q": "How do I configure account settings?",
                "a": "Account settings including user management, permissions, roles, branding, and notifications can be configured from the Settings section. There are 51 help articles covering these topics.",
                "source": "https://support.hackerrank.com/collections/4294572050-account-settings"
            },
            {
                "q": "How do I submit a support request to HackerRank?",
                "a": "Submit a support request via the HackerRank support portal at https://portal.usepylon.com/hackerrank-support/forms/customer-request-form",
                "source": "https://support.hackerrank.com/"
            },
        ]
    },

    "claude": {
        "domain_description": "Claude is Anthropic's AI assistant available via claude.ai (consumer web/mobile), Claude API, Claude Code, Claude Desktop, and enterprise/team plans.",
        "categories": [
            "Claude.ai / General Usage",
            "Pro and Max Plans / Billing",
            "Team and Enterprise Plans",
            "Claude API and Console",
            "Claude Code",
            "Claude Desktop",
            "Claude Mobile Apps",
            "Connectors / Integrations",
            "Privacy and Legal",
            "Safeguards / Safety",
            "Identity Management (SSO/SCIM)",
            "Amazon Bedrock",
            "Claude for Education",
            "Claude for Nonprofits",
            "Claude for Government",
            "Claude in Chrome",
        ],
        "escalate_triggers": [
            "account hacked",
            "unauthorized charge",
            "billing dispute",
            "data breach",
            "GDPR deletion request",
            "CCPA",
            "legal hold",
            "abuse report",
            "child safety",
            "self-harm",
            "threats",
            "law enforcement",
            "subpoena",
            "enterprise contract",
            "SOC2",
            "BAA",
            "government",
        ],
        "faq": [
            {
                "q": "What Claude plans are available?",
                "a": "Claude offers: Free (basic access), Pro (enhanced limits, priority access), Max (highest usage limits), Team (collaborative features for businesses), and Enterprise (custom contracts, SSO, advanced security). Claude API is available separately via the Anthropic Console.",
                "source": "https://support.claude.com/en/collections/4078531-claude"
            },
            {
                "q": "How do I get support for Claude?",
                "a": "Visit https://support.claude.com/en/articles/9015913-how-to-get-support for the current support options. Support channels vary by plan type. Enterprise customers have dedicated support, while consumer users can access the help center and submit requests.",
                "source": "https://support.claude.com/en/articles/9015913-how-to-get-support"
            },
            {
                "q": "What are Claude's usage limits?",
                "a": "Usage limits depend on your plan (Free, Pro, Max, Team, Enterprise). Limits reset on a rolling basis. The exact message counts are documented in the Pro and Max Plans section of the help center. If you hit a limit, you'll be notified and can wait for the reset or upgrade.",
                "source": "https://support.claude.com/en/collections/5953830-pro-and-max-plans"
            },
            {
                "q": "How do I cancel or manage my Claude subscription?",
                "a": "Subscription management (upgrade, downgrade, cancel) is available through your Claude account settings at claude.ai. For billing issues, visit the billing section of your account. Enterprise contracts require contacting your account representative.",
                "source": "https://support.claude.com/en/collections/5953830-pro-and-max-plans"
            },
            {
                "q": "How do I use the Claude API?",
                "a": "Access the Claude API through the Anthropic Console at console.anthropic.com. You'll need an API key. Documentation is at docs.claude.com. The API supports Claude models for text, vision, and tool use. Rate limits and pricing depend on your usage tier.",
                "source": "https://support.claude.com/en/collections/5370014-claude-api-and-console"
            },
            {
                "q": "What is Claude Code?",
                "a": "Claude Code is an agentic coding tool that runs in your terminal. It can read/write files, execute commands, and help with complex coding tasks. Documentation is at docs.claude.com and there are 19 help articles in the Claude Code collection.",
                "source": "https://support.claude.com/en/collections/14445694-claude-code"
            },
            {
                "q": "What are Claude's safety and content policies?",
                "a": "Claude follows Anthropic's usage policies available at anthropic.com/aup. Claude will not generate content that violates these policies. If you believe Claude refused a legitimate request incorrectly, you can provide feedback. The Safeguards collection has 15 articles on this topic.",
                "source": "https://support.claude.com/en/collections/4078535-safeguards"
            },
            {
                "q": "How does Claude handle my data and privacy?",
                "a": "Anthropic's privacy policy is at anthropic.com/privacy. Claude conversations may be used to improve the model unless you opt out (available in settings for eligible plans). Enterprise and API customers have different data handling options. The Privacy and Legal collection has 20 articles.",
                "source": "https://support.claude.com/en/collections/4078534-privacy-and-legal"
            },
            {
                "q": "Does Claude support SSO or SCIM?",
                "a": "SSO (Single Sign-On), JIT provisioning, and SCIM are available for Team and Enterprise plans. The Identity Management collection has 5 articles covering setup with common identity providers.",
                "source": "https://support.claude.com/en/collections/17270717-identity-management-sso-jit-scim"
            },
            {
                "q": "What is Claude for Teams?",
                "a": "Claude for Teams provides collaborative features for businesses: shared workspaces, team management, higher usage limits, and priority support. It's designed for teams that want to use Claude together with centralized billing.",
                "source": "https://support.claude.com/en/collections/9387370-team-and-enterprise-plans"
            },
            {
                "q": "Is Claude available on Amazon Bedrock?",
                "a": "Yes, Claude models are available on Amazon Bedrock. The Bedrock collection has 6 articles covering setup, model availability, and configuration. Access is managed through AWS, not directly through Anthropic's console.",
                "source": "https://support.claude.com/en/collections/4078537-amazon-bedrock"
            },
            {
                "q": "What connectors does Claude support?",
                "a": "Claude supports connectors for external integrations (e.g., Google Drive, etc.). There are 19 articles in the Connectors collection. Connectors allow Claude to access and work with files and data from connected services.",
                "source": "https://support.claude.com/en/collections/15399129-connectors"
            },
            {
                "q": "What is Claude in Chrome?",
                "a": "Claude in Chrome is a browser extension that allows you to use Claude while browsing the web. It can help summarize pages, answer questions about content you're viewing, and assist with tasks directly in your browser. There are 5 articles in the Claude in Chrome collection.",
                "source": "https://support.claude.com/en/collections/18031491-claude-in-chrome"
            },
            {
                "q": "Is Claude available for nonprofits or education?",
                "a": "Yes. Claude for Nonprofits (6 articles) and Claude for Education (4 articles) are available with specific programs for qualifying organizations. Visit the help center collections for eligibility and application details.",
                "source": "https://support.claude.com/en/"
            },
            {
                "q": "What is Claude Desktop?",
                "a": "Claude Desktop is a native application for macOS and Windows that provides access to Claude with system-level integrations, including MCP (Model Context Protocol) server support for connecting to local tools and data. There are 9 help articles in the Claude Desktop collection.",
                "source": "https://support.claude.com/en/collections/16163169-claude-desktop"
            },
            {
                "q": "How do I request GDPR data deletion or privacy rights?",
                "a": "For GDPR data deletion requests, data access, or other privacy rights, please contact Anthropic's privacy team via the Privacy and Legal section at https://support.claude.com/en/collections/4078534-privacy-and-legal. Anthropic's privacy policy is at anthropic.com/privacy. This requires human review and cannot be self-served.",
                "source": "https://support.claude.com/en/collections/4078534-privacy-and-legal"
            },
            {
                "q": "How do I give feedback when Claude refuses a legitimate request?",
                "a": "If Claude declined a request you believe was legitimate, you can use the thumbs-down button on the response to send feedback to Anthropic. You can also rephrase your request with more context, or review Claude's usage policies at anthropic.com/aup to understand what types of content Claude can help with. The Safeguards help collection at https://support.claude.com/en/collections/4078535-safeguards explains Claude's content policies.",
                "source": "https://support.claude.com/en/collections/4078535-safeguards"
            },
            {
                "q": "What are Claude's release notes?",
                "a": "Current release notes are at https://support.claude.com/en/articles/12138966-release-notes and cover recent changes to Claude models and products.",
                "source": "https://support.claude.com/en/articles/12138966-release-notes"
            },
        ]
    },

    "visa": {
        "domain_description": "Visa is a global payments network. Consumer support covers lost/stolen cards, disputes, declines, fraud, ATM locator, 3-D Secure/Verified by Visa, travel, and merchant issues. Visa does NOT manage individual accounts — those are handled by the card-issuing bank.",
        "categories": [
            "Lost or Stolen Card",
            "Card Declined",
            "Dispute / Chargeback",
            "Fraud / Scam",
            "ATM / Cash",
            "Travel Support",
            "3-D Secure / Verified by Visa",
            "Merchant Issues",
            "Account / Billing (issuer)",
            "Identity Theft",
            "General Information",
        ],
        "escalate_triggers": [
            "lost card",
            "stolen card",
            "unauthorized transaction",
            "identity theft",
            "fraud",
            "emergency card replacement",
            "scam",
            "account compromise",
            "data breach",
        ],
        "important_note": "Visa does not set up, service, or have access to cardholder or merchant accounts. These are managed by the issuing financial institution (bank). For account-specific issues, cardholders must contact their issuing bank using the number on the back of their card.",
        "emergency_contact": "Report a lost or stolen card: Call 000-800-100-1219 (India) or +1 303 967 1096 (reverse charge, worldwide) or +1 800 847 2911 (USA freephone).",
        "faq": [
            {
                "q": "How do I log in to my account or pay my bill?",
                "a": "To log in to your credit card account, please visit your issuer or bank's website. The website and phone number are located on the front or back of your Visa card. Visa does not manage individual cardholder accounts — this is handled by your issuing bank.",
                "source": "https://www.visa.co.in/support.html"
            },
            {
                "q": "How do I dispute a charge?",
                "a": "To dispute a charge, contact your issuer or bank using the phone number on the front or back of your Visa card. Your bank will require detailed information about the transaction before resolving the dispute. Visa does not directly process disputes — this is done through your issuing bank.",
                "source": "https://www.visa.co.in/support.html"
            },
            {
                "q": "Why was my card declined?",
                "a": "Contact your issuer or bank using the phone number on your Visa card to understand why your card was declined. Your card may be declined for various reasons (insufficient funds, security hold, incorrect PIN, expired card, etc.). Your bank is best equipped to explain the specific reason.",
                "source": "https://www.visa.co.in/support.html"
            },
            {
                "q": "What should I do if my Visa card is lost or stolen?",
                "a": "Report your lost or stolen Visa card immediately by visiting https://www.visa.co.in/support/consumer/lost-stolen-card.html or calling 000-800-100-1219 (India) or +1 303 967 1096 (worldwide reverse charge). Visa's support team can assist with reporting, blocking the card, and arranging emergency replacement services if applicable.",
                "source": "https://www.visa.co.in/support.html"
            },
            {
                "q": "How do I find an ATM?",
                "a": "Use Visa's ATM locator at https://www.visa.com/atmlocator/ to find over 2 million ATMs worldwide that accept Visa cards.",
                "source": "https://www.visa.co.in/support.html"
            },
            {
                "q": "I was contacted by someone claiming to be from Visa. Is this real?",
                "a": "This is likely a scam. Visa does NOT call or email cardholders to request personal information. Do not provide any information to the caller. You can report phone scams using Visa's name at https://www.visa.co.in/contact-us.html",
                "source": "https://www.visa.co.in/support.html"
            },
            {
                "q": "My identity has been stolen — what should I do?",
                "a": "If the identity theft involves your Visa card, visit https://www.visa.co.in/support/consumer/lost-stolen-card.html to learn how to cancel your card or get an emergency replacement. Also contact your issuing bank immediately.",
                "source": "https://www.visa.co.in/support.html"
            },
            {
                "q": "What do I need to know before travelling abroad?",
                "a": "Visit Visa's Travel Support page at https://www.visa.co.in/support/consumer/travel-support.html for information about ATM availability, emergency services, and using your card overseas. Also check with your card issuer about informing them of your travel plans.",
                "source": "https://www.visa.co.in/support.html"
            },
            {
                "q": "My card was damaged while travelling. What do I do?",
                "a": "If your card is lost, stolen, damaged, or compromised while travelling, Visa will work with your financial institution to approve and expedite an emergency replacement card, usually within 1 to 3 days. Call +1 800 847 2911 (USA freephone) or your region's Visa emergency number.",
                "source": "https://www.visa.co.in/support.html"
            },
            {
                "q": "How does 3-D Secure / Verified by Visa work?",
                "a": "3-D Secure provides an additional security layer for online (e-commerce) transactions. As a consumer, you don't need to register — your issuing bank handles the authentication. It enables validation between merchant, card issuer, and consumer to confirm the rightful account owner is making the purchase.",
                "source": "https://www.visa.co.in/support.html"
            },
            {
                "q": "3-D Secure is not working. What do I do?",
                "a": "As a consumer, you should not encounter issues with 3-D Secure. If you do, contact your issuing bank for more information using the phone number on your card.",
                "source": "https://www.visa.co.in/support.html"
            },
            {
                "q": "I have concerns about a merchant where I used my Visa card.",
                "a": "You can report merchant concerns by filling out the form at https://www.visa.co.in/Forms/visa-rules.html",
                "source": "https://www.visa.co.in/support.html"
            },
            {
                "q": "Can a merchant set a minimum or maximum amount for Visa transactions?",
                "a": "Generally, merchants are NOT permitted to set minimum or maximum amounts for Visa transactions. Exception: In the USA and US territories, merchants may set a minimum of US$10 for credit cards only. If a merchant refuses your Visa debit card due to a minimum, or requires more than US$10 minimum on credit in the USA, notify your Visa card issuer.",
                "source": "https://www.visa.co.in/support.html"
            },
            {
                "q": "How do I register for Verified by Visa / 3-D Secure?",
                "a": "As a consumer, there is nothing you need to register for 3-D Secure. Your issuing bank (financial institution) handles the authentication that 3-D Secure provides automatically.",
                "source": "https://www.visa.co.in/support.html"
            },
            {
                "q": "How do I contact Visa support?",
                "a": "For lost/stolen cards: Call 000-800-100-1219 (India) or +1 303 967 1096 (worldwide reverse charge). For other inquiries: https://usa.visa.com/Forms/contact-us-form.html. For merchant concerns: https://www.visa.co.in/Forms/visa-rules.html",
                "source": "https://www.visa.co.in/support.html"
            },
        ]
    }
}

# Escalation policies
ESCALATION_RULES = {
    "hackerrank": {
        "always_escalate": [
            "billing dispute", "refund", "invoice error", "contract", "enterprise",
            "data breach", "security incident", "account hacked", "GDPR", "CCPA",
            "legal", "subpoena", "cheating investigation appeal", "candidate complaint",
            "impersonation confirmed", "API key compromise"
        ],
        "escalation_message": "This issue requires attention from a HackerRank support specialist. Please submit a request at https://portal.usepylon.com/hackerrank-support/forms/customer-request-form or contact your account manager."
    },
    "claude": {
        "always_escalate": [
            "billing dispute", "unauthorized charge", "refund", "account hacked",
            "data breach", "GDPR deletion", "CCPA", "legal", "law enforcement",
            "abuse of another user", "self-harm", "child safety", "enterprise contract",
            "BAA", "SOC2", "API key compromise"
        ],
        "escalation_message": "This issue requires direct support from Anthropic. Please visit https://support.claude.com/en/articles/9015913-how-to-get-support for current contact options."
    },
    "visa": {
        "always_escalate": [
            "lost card", "stolen card", "unauthorized transaction", "fraud",
            "identity theft", "emergency card", "scam victim", "account compromise"
        ],
        "escalation_message": "URGENT: For lost/stolen cards or fraud, call Visa immediately at 000-800-100-1219 (India) or +1 303 967 1096 (worldwide). For account-specific issues, contact your issuing bank using the number on the back of your card."
    }
}
