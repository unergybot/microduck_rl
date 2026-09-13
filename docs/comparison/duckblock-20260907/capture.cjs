const { chromium } = require('/home/mcao/microduck-comparison/browser-tools/node_modules/playwright');
const fs = require('fs');
const out='/home/mcao/microduck-comparison/browser';
(async()=>{
 const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--enable-webgl','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 const page=await browser.newPage({viewport:{width:1100,height:760}});
 page.on('console', m=>console.log('CONSOLE',m.text().slice(0,150)));
 page.on('requestfailed',r=>console.log('FAILED',r.url(),r.failure()));
 page.on('pageerror',e=>console.log('PAGEERROR',e.message));
 await page.route('**/bundle/App-*.js',async route=>{
  let js=fs.readFileSync('/home/mcao/microduck-comparison/duckblock-main.js','utf8');
  js=js.replace('k=new u.MjData(de)','k=new u.MjData(de),window.__researchXml=d');
  js=js.replace('Ce.set(t);let n=k.ctrl','window.__researchSamples??=[],window.__researchSamples.length<2000&&window.__researchSamples.push({mode:P,sitFlag:F,obs:Array.from(e.obs.data),actions:Array.from(t)});Ce.set(t);let n=k.ctrl');
  await route.fulfill({contentType:'application/javascript',body:js});
 });
 await page.route('**/mujoco-Mp9KyG2b.wasm',r=>r.fulfill({path:out+'/mujoco.wasm',contentType:'application/wasm'}));
 await page.route('**/ort-wasm-simd-threaded-Cpm-ox6i.wasm',r=>r.fulfill({path:out+'/ort.wasm',contentType:'application/wasm'}));
 await page.route('**/*.stl',r=>{let f='/home/mcao/microduck-comparison/input/source/assets/'+new URL(r.request().url()).pathname.split('/').pop();return fs.existsSync(f)?r.fulfill({path:f,contentType:'application/octet-stream'}):r.continue()});
 await page.route('**/policies/*.onnx',r=>r.fulfill({path:'/home/mcao/microduck-comparison/input/source/policies/'+new URL(r.request().url()).pathname.split('/').pop(),contentType:'application/octet-stream'}));
 await page.goto('https://jecoprojects.com/duckblock/',{waitUntil:'domcontentloaded',timeout:90000});
 console.log('GOTO',await page.title());await page.waitForTimeout(10000);console.log('BODY',(await page.locator('body').innerText()).slice(0,1200));await page.screenshot({path:out+'/loaded.png'});
 await page.waitForFunction(()=>window.__duckBlocks&&window.__researchSamples?.length>=10,{},{timeout:120000});
 fs.writeFileSync(out+'/model.xml',await page.evaluate(()=>window.__researchXml));
 fs.writeFileSync(out+'/inference-samples.json',JSON.stringify(await page.evaluate(()=>window.__researchSamples)));
 fs.writeFileSync(out+'/initial-state.json',JSON.stringify(await page.evaluate(()=>window.__duckBlocks.getState())));
 await page.screenshot({path:out+'/initial.png'});
 console.log('READY',await page.evaluate(()=>window.__duckBlocks.getState()));
 const states=[];
 for(let i=0;i<45;i++){
  if(i===4)await page.evaluate(()=>window.__duckBlocks.trigger('sitToggle'));
  if(i===20)await page.evaluate(()=>window.__duckBlocks.trigger('walk'));
  states.push({second:i,...await page.evaluate(()=>({...window.__duckBlocks.getState(),height:window.__duckBlocks.getTrunkZ(),fallen:window.__duckBlocks.isFallen()}))});
  await page.screenshot({path:out+`/frame-${String(i).padStart(3,'0')}.png`});
  await page.waitForTimeout(1000);
 }
 fs.writeFileSync(out+'/inference-samples.json',JSON.stringify(await page.evaluate(()=>window.__researchSamples)));
 fs.writeFileSync(out+'/sitstand-parity.json',JSON.stringify(await page.evaluate(async()=>{let result=[];for(let s of window.__researchSamples.filter((s,i)=>i%20===0)){let obs=new window.rl.ort.Tensor('float32',new Float32Array(s.obs),[1,61]);let actions=(await window.rl.sessions.sitstand.run({obs})).actions.data;result.push({obs:s.obs,actions:Array.from(actions)});}return result})));
 fs.writeFileSync(out+'/states.json',JSON.stringify(states,null,2));
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
