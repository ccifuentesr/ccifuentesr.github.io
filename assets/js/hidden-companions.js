(() => {
  'use strict';
  const data=window.hiddenCompanionsData;
  if(!data)return;
  const epochs=data.epochs.filter(e=>e.accepted).sort((a,b)=>a.date.localeCompare(b.date));
  const label=e=>e.date.slice(0,16).replace('T',' · ');
  const grid=document.getElementById('ccf-sequence'),NS='http://www.w3.org/2000/svg';
  const [vMin,vMax]=data.display_range || [0,70];
  const minimum=Math.min(...epochs.flatMap(e=>e.ccf.filter((_,i)=>e.velocity[i]>=vMin&&e.velocity[i]<=vMax)));
  const yMin=Math.floor(minimum*10)/10-.02;
  const x=v=>36+(v-vMin)/(vMax-vMin)*294,y=v=>210-(v-yMin)/(1.04-yMin)*176;
  const yTicks=Array.from({length:Math.round((1-(yMin+.02))*10)+1},(_,i)=>yMin+.02+i/10);
  const ticks=Array.from({length:Math.floor(vMax/20)-Math.ceil(vMin/20)+1},(_,i)=>(Math.ceil(vMin/20)+i)*20);
  function node(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text)e.textContent=text;return e;}
  epochs.forEach((epoch,index)=>{
    const fit=epoch.display_components;
    const card=node('figure','ccf-card');
    const heading=node('figcaption','ccf-card-heading');
    heading.append(node('h3','',epoch.date.slice(0,10)));
    if(epochs.filter(e=>e.date.slice(0,10)===epoch.date.slice(0,10)).length>1)heading.append(node('span','ccf-card-time',`${epoch.date.slice(11,16)} UTC`));
    heading.append(node('span','ccf-card-instrument',epoch.instrument || 'FIES'));
    card.append(heading);
    const svg=document.createElementNS(NS,'svg');svg.setAttribute('viewBox','0 0 350 234');svg.setAttribute('role','img');svg.setAttribute('aria-label',`${data.target}, ${label(epoch)} UTC: ${fit?'two fitted components':'unresolved observed CCF'}`);
    function el(tag,attrs,text,parent=svg){const e=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,String(v)));if(text!==undefined)e.textContent=text;parent.append(e);return e;}
    el('title',{},`${data.target} · ${label(epoch)} UTC`);
    yTicks.forEach(v=>{el('line',{x1:36,x2:330,y1:y(v),y2:y(v),stroke:'var(--border-color)'});el('text',{x:27,y:y(v)+4,'text-anchor':'end','font-size':12},v.toFixed(1));});
    ticks.forEach(v=>el('text',{x:x(v),y:229,'font-size':12,'text-anchor':'middle'},v));
    let observed='';epoch.velocity.forEach((v,i)=>{if(v>=vMin&&v<=vMax)observed+=`${observed?'L':'M'}${x(v).toFixed(2)},${y(epoch.ccf[i]).toFixed(2)} `;});
    const defs=el('defs',{});
    const gradient=el('linearGradient',{id:`ccf-blend-${index}`,x1:'0%',x2:'100%'},undefined,defs);
    el('stop',{offset:'0%','stop-color':'var(--component-a)'},undefined,gradient);el('stop',{offset:'100%','stop-color':'var(--component-b)'},undefined,gradient);
    el('path',{d:observed,fill:'none',stroke:fit?'var(--text-secondary)':`url(#ccf-blend-${index})`,'stroke-width':fit?1.5:2.3,'stroke-opacity':fit?.75:1,'stroke-linecap':'round','stroke-linejoin':'round','data-observed-epoch':epoch.date});
    if(fit){
      ['a','b'].sort((first,second)=>fit[first].velocity-fit[second].velocity).forEach((key,componentIndex)=>{
        const c=fit[key];let d='';
        epoch.velocity.forEach((v,i)=>{
          if(v>=vMin&&v<=vMax&&Math.abs(v-c.velocity)<=4*c.sigma)d+=`${d?'L':'M'}${x(v).toFixed(2)},${y(epoch.ccf[i]).toFixed(2)} `;
        });
        el('path',{d,fill:'none',stroke:`var(--component-${componentIndex?'b':'a'})`,'stroke-width':2.3,'stroke-linecap':'round','stroke-linejoin':'round','data-component':componentIndex?'higher-velocity':'lower-velocity'});
      });
    }
    card.append(svg);grid.append(card);
  });

})();
