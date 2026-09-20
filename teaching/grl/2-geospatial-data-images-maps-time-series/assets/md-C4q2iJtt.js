import{r as e}from"./rolldown-runtime-hePW80VL.js";import{F as t,L as n,N as r,T as i,X as a,Y as ee,_ as o,b as s,f as c,g as l,gt as u,it as d,j as f,l as p,mt as m,ot as h,vt as g,x as _,y as v}from"./modules/shiki-C2X_BFw-.js";import{bt as y,nt as b,tt as x}from"./index-BzHpjwFJ.js";import{t as S}from"./bonn-content-D4ouftn6.js";import{t as C}from"./leaflet-CUMj7gwK.js";import{t as w}from"./lib-DQycVhIU.js";var T=e(C()),E={class:`raster-georef-task`},D={class:`task-columns`},O={class:`image-panel`},k=[`src`],A={class:`transform-panel`},j={class:`matrix-equation`,"aria-label":`Editable affine transformation matrix`},te={class:`matrix-bracket`},M={class:`matrix-row`},N={class:`matrix-row`},P={class:`state-line`},F={class:`error-line`,role:`alert`},I={class:`map-panel`},L=`+proj=utm +zone=32 +datum=WGS84 +units=m +no_defs +type=crs`,R=47,z=46,B=`❯ gdalinfo Sentinel2_RGB_cloudfree_EPSG32632.tif
Driver: GTiff/GeoTIFF
Files: Sentinel2_RGB_cloudfree_EPSG32632.tif
Size is 47, 46
Coordinate System is:
PROJCRS["WGS 84 / UTM zone 32N",
    BASEGEOGCRS["WGS 84",
        DATUM["World Geodetic System 1984",
            ELLIPSOID["WGS 84",6378137,298.257223563,
                LENGTHUNIT["metre",1]]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["degree",0.0174532925199433]],
        ID["EPSG",4326]],
    CONVERSION["UTM zone 32N",
        METHOD["Transverse Mercator",
            ID["EPSG",9807]],
        PARAMETER["Latitude of natural origin",0,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8801]],
        PARAMETER["Longitude of natural origin",9,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8802]],
        PARAMETER["Scale factor at natural origin",0.9996,
            SCALEUNIT["unity",1],
            ID["EPSG",8805]],
        PARAMETER["False easting",500000,
            LENGTHUNIT["metre",1],
            ID["EPSG",8806]],
        PARAMETER["False northing",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8807]]],
    CS[Cartesian,2],
        AXIS["(E)",east,
            ORDER[1],
            LENGTHUNIT["metre",1]],
        AXIS["(N)",north,
            ORDER[2],
            LENGTHUNIT["metre",1]],
    USAGE[
        SCOPE["Navigation and medium accuracy spatial referencing."],
        AREA["Between 6°E and 12°E, northern hemisphere between equator and 84°N, onshore and offshore. Algeria. Austria. Cameroon. Denmark. Equatorial Guinea. France. Gabon. Germany. Italy. Libya. Liechtenstein. Monaco. Netherlands. Niger. Nigeria. Norway. Sao Tome and Principe. Svalbard. Sweden. Switzerland. Tunisia. Vatican City State."],
        BBOX[0,6,84,12]],
    ID["EPSG",32632]]
Data axis to CRS axis mapping: 1,2
Origin = (365110.000000000000000,5621210.000000000000000)
Pixel Size = (10.000000000000000,-10.000000000000000)
Metadata:
  AREA_OR_POINT=Area
Image Structure Metadata:
  LAYOUT=COG
  COMPRESSION=LZW
  INTERLEAVE=PIXEL
Corner Coordinates:
Upper Left  (  365110.000, 5621210.000) (  7d 5'19.61"E, 50d43'37.24"N)
Lower Left  (  365110.000, 5620750.000) (  7d 5'20.21"E, 50d43'22.35"N)
Upper Right (  365580.000, 5621210.000) (  7d 5'43.57"E, 50d43'37.63"N)
Lower Right (  365580.000, 5620750.000) (  7d 5'44.17"E, 50d43'22.74"N)
Center      (  365345.000, 5620980.000) (  7d 5'31.89"E, 50d43'29.99"N)
Band 1 Block=256x256 Type=UInt16, ColorInterp=Gray
  Description = red
Band 2 Block=256x256 Type=UInt16, ColorInterp=Undefined
  Description = green
Band 3 Block=256x256 Type=UInt16, ColorInterp=Undefined
  Description = blue`,V=y({__name:`RasterGeorefTaskDemo`,setup(e){let{$slidev:i,$nav:ee,$clicksContext:o,$clicks:u,$page:_,$renderContext:y,$frontmatter:x}=b(),S=new URL(`/teaching/grl/2-geospatial-data-images-maps-time-series/assets/Sentinel2_RGB_cloudfree_EPSG32632-BwIDglOw.png`,``+import.meta.url).href;w.defs(`EPSG:32632`,L);let C=[`a`,`b`,`E0`,`d`,`e`,`N0`],V=d({a:``,b:``,E0:``,d:``,e:``,N0:``}),H=h(null),U=h(``),W=h(`Fill all six entries, then transform`),G=null,K=null,q=null,J=null,Y=!0,X=null,Z=null;function Q(e,t,n){return[n.a*e+n.b*t+n.E0,n.d*e+n.e*t+n.N0]}function $(e,t){let[n,r]=w(`EPSG:32632`,`EPSG:4326`,[e,t]);if(!Number.isFinite(n)||!Number.isFinite(r))throw Error(`The transform produced invalid geographic coordinates.`);if(n<-180||n>180||r<-85.05112878||r>85.05112878)throw Error(`The transformed image cannot be displayed on the map.`);return[r,n]}function ne(){let e={};for(let t of C){if(V[t]===``)throw Error(`Fill all six matrix entries.`);let n=Number(V[t]);if(!Number.isFinite(n))throw Error(`All matrix entries must be finite numbers.`);e[t]=n}let t=e.a*e.e-e.b*e.d;if(!Number.isFinite(t)||Math.abs(t)<1e-9)throw Error(`The affine matrix is singular.`);return e}function re(e){return[$(...Q(0,0,e)),$(...Q(R,0,e)),$(...Q(R,z,e)),$(...Q(0,z,e))]}function ie(){try{let e=re(ne()),t=T.default.latLngBounds(e);if(!t.isValid())throw Error(`The transformed image bounds are invalid.`);J?.remove(),K=T.default.imageOverlay(S,t,{opacity:.9,interactive:!1,className:`nearest-neighbor-raster`}),q=T.default.polygon(e,{color:`#f2c300`,weight:2,fill:!1,interactive:!1}),J=T.default.layerGroup([K,q]),Y&&J.addTo(G),X&&(X.disabled=!1),G.invalidateSize({pan:!1}),G.fitBounds(t,{padding:[18,18],maxZoom:18,animate:!1}),U.value=``,W.value=`Image transformed and placed on the map`}catch(e){U.value=e instanceof Error?e.message:`Could not transform the image.`,W.value=K?`Previous valid transform retained`:`No image placed yet`}}async function ae(){if(await f(),!H.value||G)return;G=T.default.map(H.value,{zoomControl:!0,attributionControl:!0}).setView([50.724998,7.092192],16),T.default.tileLayer(`https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png`,{maxZoom:19,attribution:`© OpenStreetMap contributors`}).addTo(G);let e=T.default.control({position:`topright`});e.onAdd=()=>{let e=T.default.DomUtil.create(`div`,`leaflet-control raster-layer-control`);return e.innerHTML=`
      <label class="visibility-option">
        <input class="raster-visibility" type="checkbox" checked disabled>
        <span>Raster</span>
      </label>
    `,T.default.DomEvent.disableClickPropagation(e),T.default.DomEvent.disableScrollPropagation(e),X=e.querySelector(`.raster-visibility`),X.addEventListener(`change`,()=>{Y=X.checked,J&&(Y?J.addTo(G):J.remove())}),e},e.addTo(G),Z=new ResizeObserver(()=>G?.invalidateSize({pan:!1})),Z.observe(H.value)}return t(ae),r(()=>{Z?.disconnect(),G?.remove(),G=null}),(e,t)=>(n(),v(`div`,E,[l(`div`,D,[l(`section`,O,[l(`img`,{src:m(S),alt:`Sentinel-2 RGB image to georeference`},null,8,k)]),l(`section`,A,[t[15]||=l(`div`,{class:`crs-label`},`EPSG:32632 · WGS 84 / UTM zone 32N · metres`,-1),l(`form`,{novalidate:``,onSubmit:c(ie,[`prevent`,`stop`])},[l(`div`,j,[t[13]||=l(`div`,{class:`matrix-name`},`A =`,-1),l(`div`,te,[l(`div`,M,[a(l(`input`,{"onUpdate:modelValue":t[0]||=e=>V.a=e,"aria-label":`Coefficient a`,type:`number`,step:`10`,onKeydown:t[1]||=c(()=>{},[`stop`])},null,544),[[p,V.a]]),a(l(`input`,{"onUpdate:modelValue":t[2]||=e=>V.b=e,"aria-label":`Coefficient b`,type:`number`,step:`10`,onKeydown:t[3]||=c(()=>{},[`stop`])},null,544),[[p,V.b]]),a(l(`input`,{"onUpdate:modelValue":t[4]||=e=>V.E0=e,"aria-label":`Easting origin E zero`,type:`number`,step:`100`,onKeydown:t[5]||=c(()=>{},[`stop`])},null,544),[[p,V.E0]])]),l(`div`,N,[a(l(`input`,{"onUpdate:modelValue":t[6]||=e=>V.d=e,"aria-label":`Coefficient d`,type:`number`,step:`10`,onKeydown:t[7]||=c(()=>{},[`stop`])},null,544),[[p,V.d]]),a(l(`input`,{"onUpdate:modelValue":t[8]||=e=>V.e=e,"aria-label":`Coefficient e`,type:`number`,step:`10`,onKeydown:t[9]||=c(()=>{},[`stop`])},null,544),[[p,V.e]]),a(l(`input`,{"onUpdate:modelValue":t[10]||=e=>V.N0=e,"aria-label":`Northing origin N zero`,type:`number`,step:`100`,onKeydown:t[11]||=c(()=>{},[`stop`])},null,544),[[p,V.N0]])]),t[12]||=l(`div`,{class:`matrix-row fixed-row`},[l(`span`,null,`0`),l(`span`,null,`0`),l(`span`,null,`1`)],-1)])]),t[14]||=l(`div`,{class:`controls`},[l(`button`,{type:`submit`},`Transform`)],-1)],32),l(`div`,P,g(W.value),1),l(`div`,F,g(U.value),1)]),l(`section`,I,[l(`div`,{ref_key:`mapElement`,ref:H,class:`map`,"aria-label":`Leaflet map for the georeferenced Sentinel-2 image`},null,512)])]),t[16]||=l(`div`,{class:`gdal-description`},[s(` The `),l(`a`,{href:`https://gdal.org/en/stable/programs/gdalinfo.html`,target:`_blank`,rel:`noopener noreferrer`},`gdalinfo`),s(` output below provides metadata for this georeferenced image. `)],-1),l(`textarea`,{class:`gdal-output`,value:B,readonly:``,spellcheck:`false`,"aria-label":`Scrollable gdalinfo output`})]))}},[[`__scopeId`,`data-v-f56260d8`]]),H={__name:`slides.md__slidev_29`,setup(e){let{$slidev:t,$nav:r,$clicksContext:a,$clicks:s,$page:c,$renderContext:d,$frontmatter:f}=b();return a.setup(),(e,t)=>{let r=V;return n(),o(S,u(i(m(x)(m(f),28))),{default:ee(()=>[t[0]||=l(`h1`,null,`Task: Georeference this Image`,-1),_(r)]),_:1},16)}}};export{H as default};