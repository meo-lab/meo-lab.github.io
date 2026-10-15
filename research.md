---
hide_title: true
layout: single
title: Research
permalink: /research/
---

{% assign topics = site.research_topics | sort: "order" %}

<h2 class="section-label" id="research-topics">Research Topics</h2>

<div class="research-topic-list">
  {% for topic in topics %}
    {% include research-topic-card.html topic=topic %}
  {% endfor %}
</div>

<h2 class="section-label" id="publications">Publications</h2>

{% assign papers = site.publications | sort: "year" | reverse %}

<div class="paper-search">
  <input class="paper-search__input" id="paper-search" type="search" placeholder="Search by title, author, venue or year" aria-label="Search publications" autocomplete="off">
  <div class="paper-search__count" id="paper-search-count" aria-live="polite">{{ papers.size }} papers</div>
</div>

<div class="paper-cards" id="paper-list">
  {% for p in papers %}
    {% include paper-card.html paper=p %}
  {% endfor %}
</div>

<p class="paper-search__empty" id="paper-search-empty" hidden>No publications match your search.</p>

<script>
  (function () {
    var input = document.getElementById('paper-search');
    var count = document.getElementById('paper-search-count');
    var empty = document.getElementById('paper-search-empty');
    var cards = Array.prototype.slice.call(document.querySelectorAll('#paper-list .paper-card'));
    var normalize = function (text) {
      return text.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
    };
    var haystacks = cards.map(function (card) { return normalize(card.textContent); });

    input.addEventListener('input', function () {
      var terms = normalize(input.value).split(/\s+/).filter(Boolean);
      var shown = 0;
      cards.forEach(function (card, i) {
        var match = terms.every(function (term) { return haystacks[i].indexOf(term) !== -1; });
        card.hidden = !match;
        if (match) shown++;
      });
      count.textContent = terms.length ? shown + ' of ' + cards.length + ' papers' : cards.length + ' papers';
      empty.hidden = shown !== 0;
    });
  })();
</script>
