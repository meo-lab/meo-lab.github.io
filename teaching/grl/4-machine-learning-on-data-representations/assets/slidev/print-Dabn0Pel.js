import{C as e,Ct as t,E as n,U as r,V as i,_ as a,b as o,h as s,v as c,w as l,x as u,yt as d}from"../modules/shiki-Db4suhHq.js";import{S as f}from"../modules/vue-CewJOpna.js";import{Nt as p,wt as m,y as h}from"../index-CpVfFuaX.js";import{t as g}from"./NoteDisplay-C58qGLwb.js";var _={id:`page-root`},v={class:`m-4`},y={class:`mb-10`},b={class:`text-4xl font-bold mt-2`},x={class:`opacity-50`},S={class:`text-lg`},C={class:`font-bold flex gap-2`},w={class:`opacity-50`},T={key:0,class:`border-main mb-8`},E=n({__name:`print`,setup(n){let{slides:E,total:D}=h();f(`
@page {
  size: A4;
  margin-top: 1.5cm;
  margin-bottom: 1cm;
}
* {
  -webkit-print-color-adjust: exact;
}
html,
html body,
html #app,
html #page-root {
  height: auto;
  overflow: auto !important;
}
`),m({title:`Notes - ${p.title}`});let O=a(()=>E.value.map(e=>e.meta?.slide).filter(e=>e!==void 0&&e.noteHTML!==``));return(n,a)=>(i(),u(`div`,_,[c(`div`,v,[c(`div`,y,[c(`h1`,b,t(d(p).title),1),c(`div`,x,t(new Date().toLocaleString()),1)]),(i(!0),u(s,null,r(O.value,(n,r)=>(i(),u(`div`,{key:r,class:`flex flex-col gap-4 break-inside-avoid-page`},[c(`div`,null,[c(`h2`,S,[c(`div`,C,[c(`div`,w,t(n?.no)+`/`+t(d(D)),1),e(` `+t(n?.title)+` `,1),a[0]||=c(`div`,{class:`flex-auto`},null,-1)])]),l(g,{"note-html":n.noteHTML,class:`max-w-full`},null,8,[`note-html`])]),r<O.value.length-1?(i(),u(`hr`,T)):o(`v-if`,!0)]))),128))])]))}});export{E as default};