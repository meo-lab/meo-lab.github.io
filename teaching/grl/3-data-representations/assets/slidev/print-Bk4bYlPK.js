import{B as e,R as t,S as n,_t as r,g as i,h as a,p as o,v as s,w as c,x as l,xt as u,y as d}from"../modules/shiki-B6qT1Nsd.js";import{S as f}from"../modules/vue-TgaEWWun.js";import{St as p,jt as m,y as h}from"../index-CD__KKkU.js";import{t as g}from"./NoteDisplay-B65HDzKg.js";var _={id:`page-root`},v={class:`m-4`},y={class:`mb-10`},b={class:`text-4xl font-bold mt-2`},x={class:`opacity-50`},S={class:`text-lg`},C={class:`font-bold flex gap-2`},w={class:`opacity-50`},T={key:0,class:`border-main mb-8`},E=c({__name:`print`,setup(c){let{slides:E,total:D}=h();f(`
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
`),p({title:`Notes - ${m.title}`});let O=a(()=>E.value.map(e=>e.meta?.slide).filter(e=>e!==void 0&&e.noteHTML!==``));return(a,c)=>(t(),d(`div`,_,[i(`div`,v,[i(`div`,y,[i(`h1`,b,u(r(m).title),1),i(`div`,x,u(new Date().toLocaleString()),1)]),(t(!0),d(o,null,e(O.value,(e,a)=>(t(),d(`div`,{key:a,class:`flex flex-col gap-4 break-inside-avoid-page`},[i(`div`,null,[i(`h2`,S,[i(`div`,C,[i(`div`,w,u(e?.no)+`/`+u(r(D)),1),l(` `+u(e?.title)+` `,1),c[0]||=i(`div`,{class:`flex-auto`},null,-1)])]),n(g,{"note-html":e.noteHTML,class:`max-w-full`},null,8,[`note-html`])]),a<O.value.length-1?(t(),d(`hr`,T)):s(`v-if`,!0)]))),128))])]))}});export{E as default};