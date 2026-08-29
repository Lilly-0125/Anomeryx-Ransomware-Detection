import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


class PSOFeatureSelector:

    def __init__(self, n_particles=20, n_iterations=30, n_features=None):
        self.n_particles  = n_particles
        self.n_iterations = n_iterations
        self.n_features   = n_features
        self.best_features = None
        self.best_score    = 0.0

    def _fitness(self, particle, X, y):
        selected = np.where(particle > 0.5)[0]
        if len(selected) == 0:
            return 0.0
        X_sub = X[:, selected]
        clf   = RandomForestClassifier(
            n_estimators=20, random_state=42, n_jobs=-1)
        scores  = cross_val_score(clf, X_sub, y, cv=3, scoring="f1")
        penalty = len(selected) / self.n_features * 0.1
        return float(scores.mean() - penalty)

    def run(self, X, y, callback=None):
        self.n_features = X.shape[1]
        particles  = np.random.rand(self.n_particles, self.n_features)
        velocities = np.random.rand(
            self.n_particles, self.n_features) * 0.1
        personal_best        = particles.copy()
        personal_best_scores = np.array(
            [self._fitness(p, X, y) for p in particles])

        best_idx         = np.argmax(personal_best_scores)
        global_best      = personal_best[best_idx].copy()
        self.best_score  = personal_best_scores[best_idx]

        w, c1, c2 = 0.7, 1.5, 1.5

        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                r1 = np.random.rand(self.n_features)
                r2 = np.random.rand(self.n_features)
                velocities[i] = (
                    w  * velocities[i]
                    + c1 * r1 * (personal_best[i] - particles[i])
                    + c2 * r2 * (global_best       - particles[i])
                )
                particles[i] = np.clip(particles[i] + velocities[i], 0, 1)
                score = self._fitness(particles[i], X, y)
                if score > personal_best_scores[i]:
                    personal_best[i]        = particles[i].copy()
                    personal_best_scores[i] = score

            best_idx = np.argmax(personal_best_scores)
            if personal_best_scores[best_idx] > self.best_score:
                global_best     = personal_best[best_idx].copy()
                self.best_score = personal_best_scores[best_idx]

            if callback:
                callback(iteration + 1, round(self.best_score, 4))

        self.best_features = np.where(global_best > 0.5)[0]
        if len(self.best_features) == 0:
            self.best_features = np.argsort(global_best)[-10:]

        return self.best_features
