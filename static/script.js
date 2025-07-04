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

    if (Array.isArray(data) && data.length > 0) {
      results.appendChild(createMovieCard(data[0], true)); // highlight searched movie

      data.slice(1).forEach(movie => {
      results.appendChild(createMovieCard(movie));
      });
    }else if (data.error) {
      results.innerHTML = `<p class="error">${data.error}</p>`;
    } else {
      results.innerHTML = `<p class="error">Unexpected response format.</p>`;
    }

  } catch (err) {
    results.innerHTML = `<p class="error">Error: ${err.message}</p>`;
  }
};

// Create a horizontal card-style movie element
function createMovieCard(movie,isSearched = false) {
  const card = document.createElement('div');
  card.className = 'movie-card';
  if (isSearched) card.classList.add('searched-movie');
  card.innerHTML = `
    <img src="${movie.poster_path || 'https://via.placeholder.com/200x300?text=No+Image'}" alt="${movie.title}">
    <h4>${movie.title}</h4>
    ${isSearched ? '<p class="badge">You searched for this movie</p>' : ''}
  `;

  // Click to open modal
  card.onclick = () => showModal(movie);

  return card;
}

// Show modal with movie details
function showModal(movie) {
  const modal = document.getElementById('movieModal');
  const content = document.getElementById('modalContent');
  let genresList = movie.genres;
  if (Array.isArray(genresList)) {
    genresList = genresList.map(g => g.replace(/[\[\]']+/g, '').trim());
  }
  const genres = genresList.join(', ');
  content.innerHTML = `
    <span id="closeModal" class="close">&times;</span>
    <img src="${movie.poster_path || 'https://via.placeholder.com/200x300?text=No+Image'}" alt="${movie.title}">
    <div class="modal-details">
      <h2>${movie.title}</h2>
      <p><strong>Release:</strong> ${movie.release_date || 'Unknown'}</p>
      <p><strong>Runtime:</strong> ${movie.runtime ? movie.runtime + ' mins' : 'N/A'}</p>
      <p><strong>Rating:</strong> ${movie.vote_average ?? 'N/A'}</p>
      <p><strong>Genres:</strong> ${genres}</p>
      <p>${movie.overview}</p>
    </div>
  `;

  modal.style.display = 'block';

  // Close modal on click and cleanup
  document.getElementById('closeModal').onclick = function () {
    modal.style.display = 'none';
    content.innerHTML = ''; // Clear content for next use
  };
}



