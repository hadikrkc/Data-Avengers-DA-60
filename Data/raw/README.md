# Raw Data Layout

This folder stores the original datasets used in the project.

Current local dataset folders:

- `IMDb Dataset`
- `Rotten Tomatoes movies and critic reviews dataset`
- `The Movies Dataset`
- `TMDB 5000 Movies Dataset`

Dataset sources:

- `The Movies Dataset`: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
- `TMDB 5000 Movies Dataset`: https://www.kaggle.com/datasets/muhammadnaumank/tmdb-5000-movies-dataset
- `Rotten Tomatoes movies and critic reviews dataset`: https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset
- `IMDb Dataset`: https://www.kaggle.com/datasets/ashirwadsangwan/imdb-dataset

Guidelines:

- Keep raw files as close to the downloaded source as possible.
- Do not commit dataset files to Git.
- If a folder name is changed, update any notebook or script paths that reference it.
- Prefer adding cleaned or renamed outputs to `Data/processed/` instead of modifying raw files in place.
