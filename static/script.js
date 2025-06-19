document.addEventListener('DOMContentLoaded', () => {
  fetchMovies('/popular', 'popular');
  fetchMovies('/latest', 'latest');
});

// Fetch movies for a section
function fetchMovies(endpoint, targetId) {
  fetch(endpoint)
    .then(res => res.json())
    .then(data => {
      const container = document.getElementById(targetId);
      container.innerHTML = '';
      data.forEach(movie => {
        container.appendChild(createMovieCard(movie));
      });
    })
    .catch(err => {
      document.getElementById(targetId).innerHTML = `<p class="error">Failed to load movies: ${err.message}</p>`;
    });
}

// Handle recommendations form submission
document.getElementById('form').onsubmit = async function (e) {
  e.preventDefault();
  const title = document.getElementById('title').value.trim();
  const results = document.getElementById('results');

  if (!title) {
    results.innerHTML = `<p class="error">Please enter a movie title.</p>`;
    return;
  }

  results.innerHTML = '<p>Loading...</p>';

  try {
    const res = await fetch('/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title })
    });

    const data = await res.json();
    results.innerHTML = '';

    if (Array.isArray(data)) {
      data.forEach(movie => {
        results.appendChild(createMovieCard(movie));
      });
    } else if (data.error) {
      results.innerHTML = `<p class="error">${data.error}</p>`;
    } else {
      results.innerHTML = `<p class="error">Unexpected response format.</p>`;
    }

  } catch (err) {
    results.innerHTML = `<p class="error">Error: ${err.message}</p>`;
  }
};

// Create a horizontal card-style movie element
function createMovieCard(movie) {
  const card = document.createElement('div');
  card.className = 'movie-card';
  card.innerHTML = `
    <img src="${movie.poster_path || 'https://via.placeholder.com/200x300?text=No+Image'}" alt="${movie.title}">
    <h4>${movie.title}</h4>
  `;

  // Click to open modal
  card.onclick = () => showModal(movie);

  return card;
}

// Show modal with movie details
function showModal(movie) {
  const modal = document.getElementById('movieModal');
  const content = document.getElementById('modalContent');
  const genres = Array.isArray(movie.genres) ? movie.genres.join(', ') : 'N/A';

  content.innerHTML = `
    <h2>${movie.title}</h2>
    <img src="${movie.poster_path || 'https://via.placeholder.com/200x300?text=No+Image'}" alt="${movie.title}">
    <p><strong>Release:</strong> ${movie.release_date || 'Unknown'}</p>
    <p><strong>Runtime:</strong> ${movie.runtime ? movie.runtime + ' mins' : 'N/A'}</p>
    <p><strong>Rating:</strong> ${movie.vote_average ?? 'N/A'}</p>
    <p><strong>Genres:</strong> ${genres}</p>
    <p>${movie.overview}</p>
  `;

  modal.style.display = 'block';
}

// Close modal on click
document.getElementById('closeModal').onclick = function () {
  document.getElementById('movieModal').style.display = 'none';
};
