# delta_h_scan
Scan recent conversation history and compute delta_H (cognitive collapse risk) across all messages.

## When to use
When Paul asks "am I overwhelmed", "check my stress level", "delta_H scan", or "how's my mental load".

## Tags
delta_h, stress, cfd, agothe, collapse, monitor, cognitive

## Parameters (via context)
- context['history'] -- list of {role, content} message dicts

## Output
Returns delta_H scores per message and an overall assessment.
