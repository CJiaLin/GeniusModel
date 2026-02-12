const API_BASE = 'http://127.0.0.1:8000/api/v1';

async function request(path, method = 'GET', data = null) {
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: data ? JSON.stringify(data) : null,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || 'Request failed');
  }
  return res.json();
}

function toObj(form) {
  const obj = Object.fromEntries(new FormData(form).entries());
  if (obj.generations) obj.generations = Number(obj.generations);
  if (obj.budget_minutes) obj.budget_minutes = Number(obj.budget_minutes);
  return obj;
}

document.getElementById('project-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const data = toObj(e.target);
  const out = document.getElementById('project-result');
  try {
    const project = await request('/projects', 'POST', data);
    out.textContent = JSON.stringify(project, null, 2);
    document.querySelector('#run-form input[name="project_id"]').value = project.id;
  } catch (err) {
    out.textContent = err.message;
  }
});

document.getElementById('run-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const data = toObj(e.target);
  const out = document.getElementById('run-result');
  try {
    const run = await request('/runs', 'POST', data);
    out.textContent = JSON.stringify(run, null, 2);
    document.querySelector('#report-form input[name="run_id"]').value = run.id;
  } catch (err) {
    out.textContent = err.message;
  }
});

document.getElementById('report-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const { run_id } = toObj(e.target);
  const out = document.getElementById('report-result');
  try {
    const report = await request(`/runs/${run_id}/report`);
    out.textContent = JSON.stringify(report, null, 2);
  } catch (err) {
    out.textContent = err.message;
  }
});
