import asyncio, json
from playwright.async_api import async_playwright
import os
URL = "file://" + os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "index.html"))
SKIP = {"reset-demo"}
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(viewport={"width": 1440, "height": 900})
        errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.route("**/fonts.*/**", lambda r: r.abort())
        await pg.goto(URL); await pg.evaluate("localStorage.clear()"); await pg.reload(); await pg.wait_for_timeout(400)
        await pg.goto(URL + "#workspace"); await pg.wait_for_timeout(100)
        await pg.click('[data-action="run-example"] >> nth=3'); await pg.wait_for_timeout(300)
        ok_counts = {}
        for name in ["home","dashboard","workspace","results","history","organizations","tools","descriptors","connectors","explorer","preparation","governance","model","environment","about"]:
            await pg.goto(URL + "#" + name); await pg.wait_for_timeout(150)
            n = await pg.evaluate("document.querySelectorAll('[data-action]').length")
            tested = 0
            for i in range(min(n, 25)):
                await pg.goto(URL + "#" + name); await pg.wait_for_timeout(80)
                els = await pg.query_selector_all("[data-action]")
                if i >= len(els): break
                el = els[i]; act = await el.get_attribute("data-action")
                if act in SKIP or not await el.is_visible() or await el.is_disabled(): continue
                before = len(errs)
                try: await el.click(timeout=1500)
                except Exception as e: continue
                await pg.wait_for_timeout(120)
                for sel in ['.dialog [data-x="no"]', '#drawer [data-x="close"]']:
                    x = await pg.query_selector(sel)
                    if x: await x.click(); await pg.wait_for_timeout(60)
                if len(errs) > before: print("ERROR on", name, act, errs[-1][:150])
                tested += 1
            ok_counts[name] = tested
        print("buttons clicked per page:", ok_counts)
        # specific checks
        await pg.goto(URL + "#history"); await pg.wait_for_timeout(150)
        print("history request rows:", await pg.evaluate("[...document.querySelectorAll('.tbl')[0].querySelectorAll('tbody tr')].length"), "distinct runs:", await pg.evaluate("Store.state.runs.length"))
        await pg.goto(URL + "#workspace"); await pg.click('[data-action="new-run"]'); await pg.wait_for_timeout(100)
        await pg.select_option('[data-filter="ex.size"]', "6+"); await pg.wait_for_timeout(150)
        print("6+ task examples:", await pg.evaluate("document.querySelectorAll('.excard').length"))
        await pg.click('[data-action="run-example"] >> nth=0'); await pg.wait_for_timeout(400)
        print("big example:", await pg.evaluate("(r=>r.status+' tasks='+r.derivation.final.length+' match='+(JSON.stringify([...r.derivation.final].sort())===JSON.stringify([...r.derivation.recorded].sort())))(Store.state.runs[Store.state.runs.length-1])"))
        ov = await pg.evaluate("""(()=>{const n=[...document.querySelectorAll('.anode rect')].map(r=>r.getBoundingClientRect());let c=0;for(let i=0;i<n.length;i++)for(let j=i+1;j<n.length;j++){const a=n[i],b=n[j];if(a.left<b.right&&b.left<a.right&&a.top<b.bottom&&b.top<a.bottom)c++;}return c;})()""")
        print("overlapping task nodes in diagram:", ov)
        await pg.screenshot(path="/tmp/c_big.png", full_page=False)
        await pg.click('[data-action="compose"]'); await pg.wait_for_timeout(300)
        print("toast after compose again:", await pg.evaluate("[...document.querySelectorAll('.toast')].map(t=>t.textContent).join(' | ')"))
        await pg.goto(URL + "#model"); await pg.wait_for_timeout(300)
        print("model status:", await pg.inner_text("#llm-status"))
        await pg.check('input[name="llm-mode"][value="openai"]'); await pg.wait_for_timeout(200)
        await pg.fill("#f-llm-base", "http://127.0.0.1:9/v1"); await pg.fill("#f-llm-key", "test"); await pg.fill("#f-llm-mname", "m"); await pg.click('[data-action="llm-test"]'); await pg.wait_for_timeout(1500)
        print("openai test (unreachable endpoint):", await pg.inner_text("#llm-test-out"))
        await pg.check('input[name="llm-mode"][value="anthropic"]'); await pg.wait_for_timeout(200); await pg.fill("#f-llm-key", ""); await pg.click('[data-action="llm-test"]'); await pg.wait_for_timeout(400)
        print("anthropic test without key:", await pg.inner_text("#llm-test-out"))
        await pg.screenshot(path="/tmp/c_model.png")
        names = await pg.evaluate("Store.state.organizations.map(o=>o.name)")
        print(len(names), "organizations; suspicious names:", [n for n in names if len(n) > 60 or n.isupper() or ' Of ' in n][:5])
        print("ERRORS:", errs[:5]); await b.close()
asyncio.run(main())
