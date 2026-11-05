(function () {
  const search = document.getElementById('permissionSearch');
  const list = document.getElementById('permissionList');
  const count = document.getElementById('selectedCount');
  if (!search || !list || !count) return;

  const items = Array.from(list.querySelectorAll('.permission-item'));
  const boxes = items.map(item => item.querySelector('input[type="checkbox"]'));

  function updateCount() {
    const total = boxes.filter(box => box.checked).length;
    count.textContent = `${total} seleccionados`;
  }

  search.addEventListener('input', () => {
    const query = search.value.trim().toLowerCase();
    items.forEach(item => {
      item.classList.toggle('hidden', !item.dataset.permission.includes(query));
    });
  });

  boxes.forEach(box => box.addEventListener('change', updateCount));

  document.getElementById('selectVisible').addEventListener('click', () => {
    items.forEach((item, index) => {
      if (!item.classList.contains('hidden')) boxes[index].checked = true;
    });
    updateCount();
  });

  document.getElementById('clearAll').addEventListener('click', () => {
    boxes.forEach(box => { box.checked = false; });
    updateCount();
  });

  updateCount();
})();
