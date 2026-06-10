

(function () {
  'use strict';

  const STYLO_KEYS = [
    'yules_k', 'honore_r', 'mattr', 'burstiness',
    'sentence_start_diversity', 'punct_entropy', 'function_word_deviation',
    'avg_sentence_length', 'type_token_ratio',
  ];

  const W_BASE = { text: 0.6, behav: 0.2, stat: 0.2 };
  const W_PROFILE = { text: 0.5, behav: 0.2, stat: 0.3 };

  const ALPHA = { copy: 0.6, pause: 0.2, edit: 0.2 };
  const MIN_PROFILE_WORKS = 5;
  const MIN_STAT_WORDS = 200;

  function wordCount(text) {
    return (String(text || '').match(/[^\s]+/g) || []).length;
  }

  function styloVector(textOrFeatures) {
    const f = typeof textOrFeatures === 'string'
      ? extractFeatures(textOrFeatures)
      : textOrFeatures;
    return STYLO_KEYS.map(k => Number(f[k]) || 0);
  }

  function corpusStats(vectors) {
    const n = vectors.length, d = STYLO_KEYS.length;
    const mean = new Array(d).fill(0);
    vectors.forEach(v => v.forEach((x, i) => { mean[i] += x; }));
    for (let i = 0; i < d; i++) mean[i] /= (n || 1);
    const std = new Array(d).fill(0);
    vectors.forEach(v => v.forEach((x, i) => { std[i] += (x - mean[i]) ** 2; }));
    for (let i = 0; i < d; i++) std[i] = Math.sqrt(std[i] / (n || 1)) || 1;
    return { mean, std };
  }

  function zNorm(vec, stats) {
    return vec.map((x, i) => (x - stats.mean[i]) / stats.std[i]);
  }

  function cosine(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    if (na === 0 || nb === 0) return 0;
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
  }

  function allCorpusWorks() {
    const out = [];
    const map = (typeof STUDENT_WORKS !== 'undefined') ? STUDENT_WORKS : {};
    Object.values(map).forEach(arr => (arr || []).forEach(w => {
      if (w && w.text && wordCount(w.text) >= MIN_STAT_WORDS) out.push(w);
    }));
    return out;
  }

  function buildProfile(studentName, subject, opts) {
    opts = opts || {};
    const excludeId = opts.excludeId || null;
    const maxAi = (typeof opts.maxAi === 'number') ? opts.maxAi : 50;
    const allWorks = allCorpusWorks();
    if (allWorks.length < 2) return { ok: false, reason: 'no_corpus', count: 0 };

    const allVecs = allWorks.map(w => styloVector(w.text));
    const stats = corpusStats(allVecs);

    const candidates = allWorks.filter(w =>
      w.student === studentName && w.subject === subject && w.id !== excludeId);

    let aiFiltered = 0;
    const studentWorks = candidates.filter(w => {
      if (typeof localAIAnalysis !== 'function') return true;
      try {
        const a = localAIAnalysis(w.text);
        if (a && typeof a.ai_probability === 'number' && a.ai_probability >= maxAi) {
          aiFiltered++;
          return false;
        }
      } catch (_) {}
      return true;
    });

    if (studentWorks.length < MIN_PROFILE_WORKS) {
      return { ok: false, reason: 'insufficient', count: studentWorks.length, candidates: candidates.length, aiFiltered, stats };
    }

    const d = STYLO_KEYS.length;
    const zVecs = studentWorks.map(w => zNorm(styloVector(w.text), stats));
    const profileVec = new Array(d).fill(0);
    zVecs.forEach(v => v.forEach((x, i) => { profileVec[i] += x; }));
    for (let i = 0; i < d; i++) profileVec[i] /= zVecs.length;
    return { ok: true, count: studentWorks.length, candidates: candidates.length, aiFiltered, profileVec, stats };
  }

  function computeRStat(text, studentName, subject, opts) {
    const wc = wordCount(text);
    if (wc < MIN_STAT_WORDS) {
      return {
        value: 0, available: false,
        reason: `недостатня довжина тексту (${wc}/${MIN_STAT_WORDS} слів)`,
        wordCount: wc,
      };
    }
    const prof = buildProfile(studentName, subject, opts);
    if (!prof.ok) {
      return {
        value: 0, available: false,
        reason: prof.reason === 'insufficient'
          ? `недостатньо даних (${prof.count || 0}/${MIN_PROFILE_WORKS} робіт)`
          : 'немає історичного корпусу',
        count: prof.count || 0,
      };
    }
    const vCurr = zNorm(styloVector(text), prof.stats);
    const sim = cosine(vCurr, prof.profileVec);

    const rstat = Math.max(0, Math.min(1, (1 - sim) / 2));
    return {
      value: +rstat.toFixed(3), available: true,
      similarity: +sim.toFixed(3), count: prof.count, wordCount: wc,
    };
  }

  function computeRBehav(telemetry) {
    if (!telemetry || !telemetry.totalChars) {
      return { value: 0, available: false, reason: 'немає телеметрії' };
    }
    const t = telemetry;

    const fCopy = Math.min(1, (t.pastedChars || 0) / Math.max(1, t.totalChars));

    const pauses = t.pauses || [];
    const longGaps = pauses.filter(p => p > 4000).length;
    const pauseScore = pauses.length
      ? Math.min(1, (longGaps / pauses.length) / 0.5)
      : (fCopy > 0.5 ? 0.8 : 0.1);

    const editRate = (t.edits || 0) / Math.max(1, t.totalChars / 100);
    const nEdit = Math.min(1, editRate / 8);
    const value = ALPHA.copy * fCopy + ALPHA.pause * pauseScore + ALPHA.edit * nEdit;
    return {
      value: +Math.max(0, Math.min(1, value)).toFixed(3),
      available: true,
      components: { f_copy: +fCopy.toFixed(3), t_pause: +pauseScore.toFixed(3), n_edit: +nEdit.toFixed(3) },
      synthetic: false,
    };
  }

  function synthRBehav(text, rText) {
    const rnd = mulberry32(hashStr((text || '').slice(0, 40)));
    let fCopy, pauseScore, nEdit;
    if (rText >= 0.6) {
      fCopy = 0.7 + rnd() * 0.3;
      pauseScore = 0.5 + rnd() * 0.4;
      nEdit = rnd() * 0.2;
    } else if (rText >= 0.35) {
      fCopy = 0.25 + rnd() * 0.35;
      pauseScore = 0.2 + rnd() * 0.3;
      nEdit = 0.2 + rnd() * 0.3;
    } else {
      fCopy = rnd() * 0.15;
      pauseScore = rnd() * 0.25;
      nEdit = 0.3 + rnd() * 0.5;
    }
    const value = ALPHA.copy * fCopy + ALPHA.pause * pauseScore + ALPHA.edit * nEdit;
    return {
      value: +Math.max(0, Math.min(1, value)).toFixed(3),
      available: true,
      components: { f_copy: +fCopy.toFixed(3), t_pause: +pauseScore.toFixed(3), n_edit: +nEdit.toFixed(3) },
      synthetic: true,
    };
  }

  function integralRisk(opts) {
    const rText = clamp01(opts.rText);
    const hasProfile = !!(opts.rStat && opts.rStat.available);
    const W = hasProfile ? W_PROFILE : W_BASE;
    const rBehav = (opts.rBehav && opts.rBehav.available) ? clamp01(opts.rBehav.value) : 0;
    const rStat = hasProfile ? clamp01(opts.rStat.value) : 0;
    const R = W.text * rText + W.behav * rBehav + W.stat * rStat;
    return {
      R: +clamp01(R).toFixed(3),
      weights: W,
      hasProfile,
      components: { R_text: +rText.toFixed(3), R_behav: +rBehav.toFixed(3), R_stat: +rStat.toFixed(3) },
      level: riskLevel(R),
    };
  }

  function riskLevel(R) {
    if (R <= 0.30) return {
      key: 'low', label: 'Низький', color: '#2F855A', range: '0.00–0.30',
      explain: 'Характеристики роботи відповідають типовому профілю студента. Стиль письма та поведінкові показники в межах норми.',
      action: 'Додаткова перевірка не потрібна. Оцінка виставляється автоматично.',
    };
    if (R <= 0.55) return {
      key: 'moderate', label: 'Помірний', color: '#B7791F', range: '0.31–0.55',
      explain: 'Незначні відхилення від типового профілю. Можливе часткове використання зовнішніх джерел.',
      action: 'Звернути увагу на роботу. За потреби — додаткове опитування студента.',
    };
    if (R <= 0.75) return {
      key: 'elevated', label: 'Підвищений', color: '#D69E2E', range: '0.56–0.75',
      explain: 'Суттєві відхилення: зміна лексичної складності, нетиповий час або ознаки великих вставок тексту.',
      action: 'Необхідна ручна перевірка. Порівняти з попередніми роботами студента.',
    };
    return {
      key: 'high', label: 'Високий', color: '#C53030', range: '0.76–1.00',
      explain: 'Критичні ознаки: різке підвищення якості порівняно з профілем, аномальна швидкість, мінімум редагувань.',
      action: 'Обов\'язкова ручна перевірка та опитування. Розглянути повторне виконання завдання.',
    };
  }

  function clamp01(x) { x = Number(x) || 0; return x < 0 ? 0 : x > 1 ? 1 : x; }
  function hashStr(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }
  function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function esc(s) { return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

  function renderIntegralCard(integral) {
    if (!integral) return '';
    const lv = integral.level;
    const c = integral.components;
    const w = integral.weights;
    const Rpct = Math.round(integral.R * 100);
    const bar = (label, val, weight, note) => {
      const pct = Math.round(val * 100);
      const bc = val >= 0.56 ? '#C53030' : val >= 0.31 ? '#D69E2E' : '#2F855A';
      return `<div style="margin-bottom:9px">
        <div style="display:flex;justify-content:space-between;font-size:.74rem;margin-bottom:3px">
          <span style="color:var(--ink2)"><b>${label}</b> <span style="color:var(--ink3)">×${weight}</span>${note ? ` <span style="color:var(--ink3);font-size:.68rem">${esc(note)}</span>` : ''}</span>
          <span style="font-family:var(--ff);font-weight:600;color:${bc}">${val.toFixed(2)}</span>
        </div>
        <div style="height:6px;background:var(--bg3);border-radius:3px;overflow:hidden">
          <div style="width:${pct}%;height:100%;background:${bc};border-radius:3px"></div>
        </div>
      </div>`;
    };
    const statNote = integral.rStatReason ? integral.rStatReason : '';
    const behavNote = integral.rBehavSynthetic ? 'симуляція' : 'телеметрія';
    return `<div class="card" style="margin-bottom:12px;border:1.5px solid ${lv.color}44;background:linear-gradient(180deg,${lv.color}0A,transparent)">
      <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:13px">
        <div style="width:88px;height:88px;border-radius:50%;border:5px solid ${lv.color};display:flex;flex-direction:column;align-items:center;justify-content:center;flex-shrink:0">
          <div style="font-family:var(--ff);font-size:1.5rem;font-weight:600;color:${lv.color};line-height:1">${integral.R.toFixed(2)}</div>
          <div style="font-size:.58rem;color:var(--ink3);text-transform:uppercase;letter-spacing:.08em;margin-top:2px">ризик R</div>
        </div>
        <div style="flex:1;min-width:180px">
          <div style="font-size:.66rem;color:var(--ink3);text-transform:uppercase;letter-spacing:.07em;margin-bottom:3px">Інтегральна оцінка доброчесності</div>
          <div style="font-family:var(--ff);font-size:1.25rem;font-weight:600;color:${lv.color};line-height:1.2">${lv.label} ризик</div>
          <div style="font-size:.72rem;color:var(--ink3);margin-top:2px">діапазон ${lv.range} · R = ${Rpct}%</div>
        </div>
      </div>

      <div style="background:var(--bg2);border-radius:var(--r2);padding:11px 13px;margin-bottom:11px">
        ${bar('R_text (аналіз тексту)', c.R_text, w.text, '')}
        ${bar('R_behav (поведінка)', c.R_behav, w.behav, behavNote)}
        ${bar('R_stat (відхилення стилю)', c.R_stat, w.stat, statNote)}
        <div style="font-size:.68rem;color:var(--ink3);margin-top:7px;padding-top:7px;border-top:1px solid var(--line);font-family:var(--ff)">
          R = ${w.text}·${c.R_text.toFixed(2)} + ${w.behav}·${c.R_behav.toFixed(2)} + ${w.stat}·${c.R_stat.toFixed(2)} = <b style="color:${lv.color}">${integral.R.toFixed(3)}</b>
          ${integral.hasProfile ? ' <span style="color:var(--ink3)">(режим профілю)</span>' : ' <span style="color:var(--ink3)">(базовий режим)</span>'}
        </div>
      </div>

      <div style="font-size:.78rem;color:var(--ink2);line-height:1.55;margin-bottom:6px">${esc(lv.explain)}</div>
      <div style="font-size:.76rem;line-height:1.5;padding:8px 11px;background:${lv.color}10;border-left:3px solid ${lv.color};border-radius:4px">
        <b style="color:${lv.color}">Рекомендація:</b> ${esc(lv.action)}
      </div>
    </div>`;
  }

  const api = {
    STYLO_KEYS, MIN_PROFILE_WORKS, ALPHA, W_BASE, W_PROFILE,
    styloVector, buildProfile, computeRStat, computeRBehav, synthRBehav,
    integralRisk, riskLevel, renderIntegralCard,
  };
  if (typeof window !== 'undefined') window.Integrity = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})();
