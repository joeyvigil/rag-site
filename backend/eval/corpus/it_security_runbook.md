# IT Security Runbook

Use the incident code in the subject line of every report.

## SEC-417: Suspected phishing email

Do not click any links or open attachments. Forward the email as an attachment to
phish@halcyon-robotics.example and then delete it. The security team responds within
4 business hours.

## SEC-502: Lost or stolen device

Report the loss within 24 hours by filing a ticket in the HW-LOSS queue. IT will remotely
wipe the device. If the device contained customer data, also notify your manager.

## SEC-610: Production credentials exposed

Rotate the credentials immediately, then page the on-call security engineer via the
#sec-oncall channel. Severity is always P1. Do not delete logs.

## VPN error codes

- ERR_VPN_0x2F: certificate expired. Run `halcyon-vpn renew-cert` and reconnect.
- ERR_VPN_0x31: MFA timeout. Approve the push notification within 60 seconds.
- ERR_VPN_0x44: Account locked after 5 failed attempts. Contact the ServiceDesk to unlock it.

## Password policy

Passwords must be at least 14 characters long and are rotated every 180 days.
Multi-factor authentication (MFA) is mandatory for all accounts.
