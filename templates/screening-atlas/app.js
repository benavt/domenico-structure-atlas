'use strict';
const $=id=>document.getElementById(id);let atlas,viewer,active=0,serial=0;
const titles={median_matrix:['Median Cα distance','Median pairwise distances across all accepted models, in Å.'],variance_matrix:['Cα distance variance','Saved distance variance across the full ensemble, in Å². Axes follow the same sequence order as the median matrix.'],box_summary_3d:['Pair-distance distribution overview','Saved full ensemble summary. Dense whiskers can obscure the median and IQR; use the distance explorer for individual pairs.'],pair_distribution_detail:['One saved residue-pair distribution','A diagnostic example pair from the completed screening. It does not summarize every residue pair.']};
async function loadJSON(path){const r=await fetch(path);if(!r.ok)throw Error('Could not load saved results ('+r.status+').');return r.json()}
function fail(error){$('error').hidden=false;$('error').textContent=error.message;console.error(error)}
async function selectTarget(id){
 const target=atlas.targets.find(t=>t.id===id)||atlas.targets[0],request=++serial;active=target.id;
 for(const button of $('targets').children)button.setAttribute('aria-current',String(button.dataset.id===target.id));
 history.replaceState(null,'','#'+target.id);$('error').hidden=true;
 $('targetId').textContent=target.id;$('targetTitle').textContent=target.title;
 $('targetSummary').textContent=target.models.toLocaleString()+' models · '+target.methods.length+' methods · '+target.residues+' residues in the supplied construct';
 $('rangeText').textContent='Chain A · construct positions 1–'+target.residues+' · all retained';
 $('scoresDownload').href=target.scores_csv;$('terminalDownload').href=target.terminal_json;$('explorerLink').href=target.distance_explorer;
 $('coverageText').textContent=(target.complete_method_panel?'Complete eleven-method panel. ':'Partial method panel in this fixed snapshot. ')+Object.entries(target.physicality).map(([key,n])=>n.toLocaleString()+' '+key).join(' · ')+'.';
 $('uniprotLink').href='https://www.uniprot.org/uniprotkb/'+target.id.split('_').at(-1)+'/entry';
 $('targetProvenance').textContent='Accepted manifest SHA-256: '+atlas.source_manifest_sha256+'. Screening SHA-256: '+target.screening_sha256+'. Exact retained-pair mask: '+target.pair_mask_sha256+'.';
 $('exampleLabel').textContent='AF-Cluster · '+target.example.model+' · illustrative input, not a ranked or cluster representative. Drag to rotate; scroll to zoom.';
 $('exampleDownload').href=target.example.file;
 $('figures').replaceChildren(...Object.entries(target.images).map(([key,path])=>{const figure=document.createElement('figure'),link=document.createElement('a'),image=document.createElement('img'),caption=document.createElement('figcaption'),title=document.createElement('b');link.href=path;link.target='_blank';link.rel='noopener';image.src=path;image.alt=titles[key][0]+' for '+target.title;image.loading='lazy';link.append(image);title.textContent=titles[key][0];caption.append(title,document.createTextNode(titles[key][1]));figure.append(link,caption);return figure}));
 const scores=await loadJSON(target.scores);if(request!==serial)return;
 $('methodRows').replaceChildren(...target.methods.map(method=>{const tr=document.createElement('tr'),name=document.createElement('td'),dot=document.createElement('span'),count=document.createElement('td'),flag=document.createElement('td');dot.className='method-dot';dot.style.background=method.color;name.append(dot,document.createTextNode(method.name));count.textContent=method.count.toLocaleString();const rejected=scores.filter(r=>r.method===method.key&&r.native_physicality==='rejected').length;flag.textContent=rejected?rejected.toLocaleString()+' retained':'0';tr.append(name,count,flag);return tr}));
 const traces=target.methods.map(method=>({type:'box',name:method.name,y:scores.filter(r=>r.method===method.key).map(r=>r.energy),boxpoints:false,marker:{color:method.color},line:{width:1.3,color:method.color},hovertemplate:'ProteinEBM t=0.10: %{y:.3f}<extra>'+method.name+'</extra>'}));
 await Plotly.react($('energyPlot'),traces,{margin:{t:12,l:70,r:12,b:100},paper_bgcolor:'#fff',plot_bgcolor:'#fff',font:{family:'Arial, sans-serif',size:10,color:'#566c74'},showlegend:false,yaxis:{title:{text:'ProteinEBM energy · arbitrary units',font:{size:10}},gridcolor:'#edf1ee',zeroline:false},xaxis:{tickangle:-48},uirevision:target.id},{responsive:true,displaylogo:false,toImageButtonOptions:{format:'svg',filename:target.id+'-saved-proteinebm-t010'}});
 if(request!==serial)return;
 const response=await fetch(target.example.file);if(!response.ok)throw Error('Illustrative saved model is unavailable.');const bytes=await response.arrayBuffer();
 if(crypto.subtle){const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join('');if(hash!==target.example.sha256)throw Error('Illustrative coordinate hash changed.');}
 if(request!==serial)return;
 viewer.removeAllModels();viewer.addModel(new TextDecoder().decode(bytes),target.example.format);viewer.setStyle({},{cartoon:{color:'#237c72'}});viewer.setBackgroundColor('#ffffff');viewer.zoomTo();viewer.render();
}
window.addEventListener('resize',()=>viewer?.resize());
$('resetView').addEventListener('click',()=>{viewer.zoomTo();viewer.render()});
async function start(){
 atlas=await loadJSON('atlas.json');viewer=$3Dmol.createViewer($('structure'),{backgroundColor:'white',antialias:true});
 $('targets').replaceChildren(...atlas.targets.map(target=>{const button=document.createElement('button'),title=document.createElement('strong'),detail=document.createElement('span');button.type='button';button.dataset.id=target.id;title.textContent=target.title;detail.textContent=target.models.toLocaleString()+' models · '+target.residues+' residues';button.append(title,detail);button.addEventListener('click',()=>selectTarget(target.id).catch(fail));return button}));
 await selectTarget(decodeURIComponent(location.hash.slice(1)));
}
start().catch(fail);
