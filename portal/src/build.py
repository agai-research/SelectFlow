import os
d = os.path.dirname(os.path.abspath(__file__)) + "/"
html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>SelectFlow</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/2.11.8/umd/popper.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/tippy.js/6.3.7/tippy-bundle.umd.min.js"></script>
<style>
{open(d+'style.css').read()}
</style>
</head>
<body>
<div class="app" id="app"></div>
<div class="toast-area" id="toasts" aria-live="polite"></div>
<script>
{open(d+'data.js').read()}
{open(d+'seed.js').read()}
{open(d+'engine.js').read()}
{open(d+'services.js').read()}
{open(d+'ui1.js').read().replace("boot();", "")}
{open(d+'ui2.js').read().replace("boot();", "")}
{open(d+'ui3.js').read()}
boot();
</script>
</body>
</html>"""
open(os.path.join(d, "..", "index.html"), "w").write(html)
print(len(html))
