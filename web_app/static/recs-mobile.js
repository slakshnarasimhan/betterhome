(function () {
  function enhance(el) {
    if (el.dataset.descReady === '1') return;
    if (el.clientHeight === 0) return;
    el.classList.add('product-desc');
    el.classList.remove('is-open');
    if (el.scrollHeight <= el.clientHeight + 2) {
      el.dataset.descReady = '1';
      return;
    }
    el.dataset.descReady = '1';
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'desc-more';
    btn.textContent = 'More';
    btn.addEventListener('click', function () {
      var open = el.classList.toggle('is-open');
      btn.textContent = open ? 'Less' : 'More';
    });
    el.insertAdjacentElement('afterend', btn);
  }

  function enhanceDescriptions() {
    document.querySelectorAll('.product-desc, .accordion-body .col-md-7 > p.mb-4').forEach(enhance);
  }

  function loadXlsx(done) {
    if (window.XLSX) return done();
    var s = document.createElement('script');
    s.src = '/static/xlsx.full.min.js';
    s.onload = done;
    s.onerror = function () { alert('Could not load Excel library.'); };
    document.head.appendChild(s);
  }

  function expandAllRooms() {
    document.querySelectorAll('.accordion-collapse').forEach(function (el) {
      el.classList.add('show');
      el.style.display = 'block';
      el.style.height = 'auto';
    });
    document.querySelectorAll('.product-desc').forEach(function (el) {
      el.classList.add('is-open');
    });
  }

  function downloadPdf() {
    expandAllRooms();
    setTimeout(function () { window.print(); }, 250);
  }

  function textOf(el) {
    return el ? (el.textContent || '').replace(/\s+/g, ' ').trim() : '';
  }

  function downloadExcel() {
    loadXlsx(function () {
      var data = [['Room', 'Category', 'Product', 'Price']];
      document.querySelectorAll('.accordion-item').forEach(function (item) {
        var room = textOf(item.querySelector('.accordion-button'));
        item.querySelectorAll('.product-block, .accordion-body .container').forEach(function (block) {
          if (!block.querySelector('h6')) return;
          var cat = '';
          var prev = block.previousElementSibling;
          if (prev && prev.tagName === 'H4') cat = textOf(prev);
          data.push([
            room,
            cat,
            textOf(block.querySelector('h6')),
            textOf(block.querySelector('.h2, .bh-price'))
          ]);
        });
      });
      data.push([]);
      data.push(['Client Information']);
      var name = textOf(document.querySelector('.contact-section h5'));
      if (name) data.push(['Name', name]);
      document.querySelectorAll('.contact-section .row').forEach(function (row) {
        var label = textOf(row.querySelector('.col-sm-3'));
        var value = textOf(row.querySelector('.col-sm-9'));
        if (label && value) data.push([label, value]);
      });
      var ws = XLSX.utils.aoa_to_sheet(data);
      var wb = XLSX.utils.book_new();
      XLSX.utils.book_append_sheet(wb, ws, 'Recommendations');
      XLSX.writeFile(wb, 'AppliancesBazaar_Recommendations.xlsx');
    });
  }

  function addExportBar() {
    if (document.getElementById('recs-export-bar')) return;
    var headings = document.querySelectorAll('h2.fw-bolder');
    var target = null;
    headings.forEach(function (h) {
      if ((h.textContent || '').indexOf('Product Recommendations') !== -1) target = h;
    });
    if (!target) return;
    var bar = document.createElement('div');
    bar.id = 'recs-export-bar';
    bar.className = 'recs-export-bar';
    bar.innerHTML =
      '<button type="button" data-action="pdf">Save as PDF</button>' +
      '<button type="button" data-action="excel">Download Excel</button>';
    bar.addEventListener('click', function (e) {
      var action = e.target && e.target.getAttribute('data-action');
      if (action === 'pdf') downloadPdf();
      if (action === 'excel') downloadExcel();
    });
    target.parentNode.appendChild(bar);
  }

  function init() {
    enhanceDescriptions();
    addExportBar();
    document.querySelectorAll('.accordion-collapse').forEach(function (panel) {
      panel.addEventListener('shown.bs.collapse', enhanceDescriptions);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
