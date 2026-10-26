import{B as e,C as t,H as n,S as r,St as i,T as a,_ as o,b as s,m as c,v as l,vt as u,x as d}from"../modules/shiki-tW7ZzN0Q.js";import{S as f}from"../modules/vue-D-4FSHFP.js";import{Ct as p,Mt as m,y as h}from"../index-xtoHZ8US.js";import{t as g}from"./NoteDisplay-D_BG8qGX.js";var _={id:`page-root`},v={class:`m-4`},y={class:`mb-10`},b={class:`text-4xl font-bold mt-2`},x={class:`opacity-50`},S={class:`text-lg`},C={class:`font-bold flex gap-2`},w={class:`opacity-50`},T={key:0,class:`border-main mb-8`},E=a({__name:`print`,setup(a){let{slides:E,total:D}=h();f(`
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
`),p({title:`Notes - ${m.title}`});let O=o(()=>E.value.map(e=>e.meta?.slide).filter(e=>e!==void 0&&e.noteHTML!==``));return(a,o)=>(e(),d(`div`,_,[l(`div`,v,[l(`div`,y,[l(`h1`,b,i(u(m).title),1),l(`div`,x,i(new Date().toLocaleString()),1)]),(e(!0),d(c,null,n(O.value,(n,a)=>(e(),d(`div`,{key:a,class:`flex flex-col gap-4 break-inside-avoid-page`},[l(`div`,null,[l(`h2`,S,[l(`div`,C,[l(`div`,w,i(n?.no)+`/`+i(u(D)),1),r(` `+i(n?.title)+` `,1),o[0]||=l(`div`,{class:`flex-auto`},null,-1)])]),t(g,{"note-html":n.noteHTML,class:`max-w-full`},null,8,[`note-html`])]),a<O.value.length-1?(e(),d(`hr`,T)):s(`v-if`,!0)]))),128))])]))}});export{E as default};