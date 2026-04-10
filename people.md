---
layout: single
title: People
permalink: /people/
---

{% assign leaders = site.members | where: "label", "leader" | sort: "order" %}
{% assign phds = site.members | where: "label", "PhD" | sort: "order" %}
{% assign visiting = site.members | where: "label", "Visiting" | sort: "name" %}
{% assign msc_all = site.members | where: "label", "MSc" %}
{% assign current_msc = msc_all | where: "current_msc", true | sort: "end" | reverse %}
{% assign former_msc = msc_all | where_exp: "m", "m.current_msc != true" | sort: "end" | reverse %}
{% assign collaborators = site.collaborators | sort: "start" | reverse %}

<div class="people-page">
  <section class="people-section people-section--leader">
    <div class="people-section__header">
      <div class="people-section__eyebrow">Research Group</div>
      <h2 class="people-section__title">Jun. Prof. &amp; Emmy Noether Group Leader</h2>
    </div>

    <div class="people-featured">
      {% for m in leaders %}
        <article class="person-card person-card--featured">
          {% if m.featured_image or m.image %}
            <div class="person-card__media person-card__media--featured">
              <img src="{{ m.featured_image | default: m.image | relative_url }}" alt="{{ m.name }}">
            </div>
          {% endif %}
          <div class="person-card__content person-card__content--featured">
            <div class="person-card__label">Group Leader</div>
            <div class="person-card__name person-card__name--featured">
              {% if m.website %}
                <a href="{{ m.website }}" target="_blank" rel="noopener">{{ m.name }}</a>
              {% else %}
                {{ m.name }}
              {% endif %}
            </div>
            <div class="person-card__meta person-card__meta--featured">{{ m.position }}</div>
            {% if m.former_positions %}
              <div class="person-card__detail person-card__detail--list">
                <span class="person-card__detail-label">Previous Positions</span>
                <ul class="person-card__list">
                  {% for former in m.former_positions %}
                    <li>{{ former }}</li>
                  {% endfor %}
                </ul>
              </div>
            {% endif %}
          </div>
        </article>
      {% endfor %}
    </div>
  </section>

  <section class="people-section">
    <div class="people-section__header">
      <div class="people-section__eyebrow">Team</div>
      <h2 class="people-section__title">Doctoral Candidates (external)</h2>
    </div>

    <div class="people-grid">
      {% for m in phds %}
        <article class="person-card">
          {% if m.image %}
            <div class="person-card__media">
              <img src="{{ m.image | relative_url }}" alt="{{ m.name }}">
            </div>
          {% endif %}
          <div class="person-card__content">
            <div class="person-card__name">
              {% if m.website %}
                <a href="{{ m.website }}" target="_blank" rel="noopener">{{ m.name }}</a>
              {% else %}
                {{ m.name }}
              {% endif %}
            </div>
            <div class="person-card__meta">{{ m.position_long | default: m.position }}</div>
            {% if m.topic or m.phd_topic %}
              <div class="person-card__detail">{{ m.topic | default: m.phd_topic }}</div>
            {% endif %}
          </div>
        </article>
      {% endfor %}
    </div>
  </section>

  {% if visiting.size > 0 %}
    <section class="people-section">
      <div class="people-section__header">
        <div class="people-section__eyebrow">Visitors</div>
        <h2 class="people-section__title">Visiting Scholars</h2>
      </div>

      <div class="people-grid">
        {% for m in visiting %}
          <article class="person-card">
            {% if m.image %}
              <div class="person-card__media">
                <img src="{{ m.image | relative_url }}" alt="{{ m.name }}">
              </div>
            {% endif %}
            <div class="person-card__content">
              <div class="person-card__name">
                {% if m.website %}
                  <a href="{{ m.website }}" target="_blank" rel="noopener">{{ m.name }}</a>
                {% else %}
                  {{ m.name }}
                {% endif %}
              </div>
              <div class="person-card__meta">{{ m.position_long | default: m.position }}</div>
              {% if m.topic or m.visiting_topic %}
                <div class="person-card__detail">{{ m.topic | default: m.visiting_topic }}</div>
              {% endif %}
            </div>
          </article>
        {% endfor %}
      </div>
    </section>
  {% endif %}

  <section class="people-section people-section--compact">
    <div class="people-section__header">
      <div class="people-section__eyebrow">Network</div>
      <h2 class="people-section__title">Collaborators</h2>
    </div>

    <div class="collaborator-line">
      {% for c in collaborators %}
        <span class="collaborator-name">
          {% if c.website %}
            <a href="{{ c.website }}" target="_blank" rel="noopener">{{ c.name }}</a>
          {% else %}
            {{ c.name }}
          {% endif %}
        </span>
        {% if c.start or c.end or c.topic or c.selected_works %}
          <span class="collaborator-meta">
            (
            {% if c.start %}{{ c.start }}{% endif %}
            {% if c.end %}-{{ c.end }}{% endif %}
            {% if c.topic %}{% if c.start or c.end %}, {% endif %}{{ c.topic }}{% endif %}
            {% if c.selected_works %}
              {% if c.start or c.end or c.topic %}, {% endif %}
              Joint works:
              {% for work in c.selected_works %}
                <a href="{{ work.url }}" target="_blank" rel="noopener">{{ work.title }}</a>{% unless forloop.last %}, {% endunless %}
              {% endfor %}
            {% endif %}
            )
          </span>
        {% endif %}
        {% unless forloop.last %}<span class="collaborator-sep"> • </span>{% endunless %}
      {% endfor %}
    </div>
  </section>

  <section class="people-section people-section--compact">
    <div class="people-section__header">
      <div class="people-section__eyebrow">Alumni</div>
      <h2 class="people-section__title">Former MSc Students</h2>
    </div>

    <div class="former-msc-line">
      {% for m in former_msc %}
        <span class="former-msc-name">
          {% if m.website %}
            <a href="{{ m.website }}" target="_blank" rel="noopener">{{ m.name }}</a>
          {% else %}
            {{ m.name }}
          {% endif %}
        </span>
        {% if m.end or m.award or m.thesis_url %}
          <span class="former-msc-meta">
            ({% if m.end %}{{ m.end }}{% endif %}
            {% if m.thesis_url %}{% if m.end %}, {% endif %}<a href="{{ m.thesis_url }}" target="_blank" rel="noopener">MSc Thesis</a>{% endif %}
            {% if m.award %}{% if m.end or m.thesis_url %}, {% endif %}{{ m.award }}{% endif %})
          </span>
        {% endif %}
        {% unless forloop.last %}<span class="former-msc-sep"> • </span>{% endunless %}
      {% endfor %}
    </div>
  </section>
</div>
