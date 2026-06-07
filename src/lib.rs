//! Multi-directional decision analysis.
//!
//! Provides tools for navigating decision spaces using directional metaphors:
//! bearings, cardinal and ordinal strategies, deviation measurement, and
//! path planning through complex decision landscapes.


// ============================================================================
// bearing module
// ============================================================================

pub mod bearing {
    /// A bearing represents the direction of a decision in an N-dimensional space.
    #[derive(Debug, Clone)]
    pub struct Bearing {
        pub components: Vec<f64>,
        pub magnitude: f64,
        pub label: Option<String>,
    }

    impl Bearing {
        pub fn new(components: Vec<f64>) -> Self {
            let magnitude = components.iter().map(|c| c * c).sum::<f64>().sqrt();
            Self { components, magnitude, label: None }
        }

        pub fn with_label(mut self, label: &str) -> Self {
            self.label = Some(label.to_string());
            self
        }

        pub fn dimensions(&self) -> usize {
            self.components.len()
        }

        pub fn is_zero(&self) -> bool {
            self.magnitude < 1e-10
        }

        pub fn normalize(&self) -> Bearing {
            if self.magnitude < 1e-10 {
                return Bearing::new(vec![0.0; self.components.len()]);
            }
            Bearing {
                components: self.components.iter().map(|c| c / self.magnitude).collect(),
                magnitude: 1.0,
                label: self.label.clone(),
            }
        }

        pub fn dot(&self, other: &Bearing) -> f64 {
            self.components.iter().zip(other.components.iter())
                .map(|(a, b)| a * b)
                .sum()
        }

        pub fn angle_to(&self, other: &Bearing) -> f64 {
            let d = self.dot(other);
            let denom = self.magnitude * other.magnitude;
            if denom < 1e-10 { return 0.0; }
            (d / denom).clamp(-1.0, 1.0).acos()
        }

        pub fn add(&self, other: &Bearing) -> Bearing {
            let max_dim = self.components.len().max(other.components.len());
            let comps: Vec<f64> = (0..max_dim).map(|i| {
                self.components.get(i).copied().unwrap_or(0.0) +
                other.components.get(i).copied().unwrap_or(0.0)
            }).collect();
            Bearing::new(comps)
        }

        pub fn scale(&self, factor: f64) -> Bearing {
            Bearing::new(self.components.iter().map(|c| c * factor).collect())
                .with_label(self.label.as_deref().unwrap_or(""))
        }

        pub fn distance_to(&self, other: &Bearing) -> f64 {
            let max_dim = self.components.len().max(other.components.len());
            (0..max_dim).map(|i| {
                let a = self.components.get(i).copied().unwrap_or(0.0);
                let b = other.components.get(i).copied().unwrap_or(0.0);
                (a - b).powi(2)
            }).sum::<f64>().sqrt()
        }

        pub fn project_onto(&self, other: &Bearing) -> Bearing {
            let d = self.dot(other);
            let other_mag_sq = other.magnitude * other.magnitude;
            if other_mag_sq < 1e-10 {
                return Bearing::new(vec![0.0; self.components.len()]);
            }
            let scalar = d / other_mag_sq;
            other.scale(scalar)
        }

        pub fn reflect(&self, normal: &Bearing) -> Bearing {
            let proj = self.project_onto(normal);
            self.add(&proj.scale(-2.0))
        }

        pub fn component(&self, index: usize) -> Option<f64> {
            self.components.get(index).copied()
        }
    }

    /// Create a unit bearing along a specific axis.
    pub fn axis_bearing(axis: usize, dimensions: usize) -> Bearing {
        let mut components = vec![0.0; dimensions];
        if axis < dimensions {
            components[axis] = 1.0;
        }
        Bearing::new(components)
    }

    /// Interpolate between two bearings.
    pub fn interpolate(a: &Bearing, b: &Bearing, t: f64) -> Bearing {
        let a_scaled = a.scale(1.0 - t);
        let b_scaled = b.scale(t);
        a_scaled.add(&b_scaled)
    }
}

// ============================================================================
// cardinal module
// ============================================================================

pub mod cardinal {
    use super::bearing::Bearing;

    /// Cardinal directions as decision strategies in 2D space.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Cardinal {
        North,
        South,
        East,
        West,
    }

    impl Cardinal {
        pub fn bearing(&self) -> Bearing {
            match self {
                Cardinal::North => Bearing::new(vec![0.0, 1.0]),
                Cardinal::South => Bearing::new(vec![0.0, -1.0]),
                Cardinal::East => Bearing::new(vec![1.0, 0.0]),
                Cardinal::West => Bearing::new(vec![-1.0, 0.0]),
            }
        }

        pub fn opposite(&self) -> Cardinal {
            match self {
                Cardinal::North => Cardinal::South,
                Cardinal::South => Cardinal::North,
                Cardinal::East => Cardinal::West,
                Cardinal::West => Cardinal::East,
            }
        }

        pub fn label(&self) -> &str {
            match self {
                Cardinal::North => "North",
                Cardinal::South => "South",
                Cardinal::East => "East",
                Cardinal::West => "West",
            }
        }

        pub fn all() -> Vec<Cardinal> {
            vec![Cardinal::North, Cardinal::East, Cardinal::South, Cardinal::West]
        }

        /// Which cardinal direction is closest to the given bearing.
        pub fn closest(bearing: &Bearing) -> Cardinal {
            let all = Cardinal::all();
            all.into_iter().max_by(|a, b| {
                let da = a.bearing().dot(bearing);
                let db = b.bearing().dot(bearing);
                da.partial_cmp(&db).unwrap()
            }).unwrap()
        }

        /// Interpret the cardinal as a decision strategy.
        pub fn strategy(&self) -> &str {
            match self {
                Cardinal::North => "Expand/Grow",
                Cardinal::South => "Consolidate/Reduce",
                Cardinal::East => "Explore/Innovate",
                Cardinal::West => "Retrench/Simplify",
            }
        }
    }

    /// Score each cardinal direction for a given context.
    pub fn score_cardinals(context: &Bearing) -> Vec<(Cardinal, f64)> {
        Cardinal::all().into_iter().map(|c| {
            let score = c.bearing().dot(context);
            (c, score)
        }).collect()
    }

    /// Find the dominant cardinal direction.
    pub fn dominant(context: &Bearing) -> Cardinal {
        Cardinal::closest(context)
    }
}

// ============================================================================
// ordinal module
// ============================================================================

pub mod ordinal {
    use super::bearing::Bearing;
    use super::cardinal::Cardinal;

    /// Ordinal (intermediate) directions.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Ordinal {
        NorthEast,
        NorthWest,
        SouthEast,
        SouthWest,
    }

    impl Ordinal {
        pub fn bearing(&self) -> Bearing {
            let s = 1.0 / 2.0_f64.sqrt();
            match self {
                Ordinal::NorthEast => Bearing::new(vec![s, s]),
                Ordinal::NorthWest => Bearing::new(vec![-s, s]),
                Ordinal::SouthEast => Bearing::new(vec![s, -s]),
                Ordinal::SouthWest => Bearing::new(vec![-s, -s]),
            }
        }

        pub fn cardinals(&self) -> (Cardinal, Cardinal) {
            match self {
                Ordinal::NorthEast => (Cardinal::North, Cardinal::East),
                Ordinal::NorthWest => (Cardinal::North, Cardinal::West),
                Ordinal::SouthEast => (Cardinal::South, Cardinal::East),
                Ordinal::SouthWest => (Cardinal::South, Cardinal::West),
            }
        }

        pub fn label(&self) -> &str {
            match self {
                Ordinal::NorthEast => "NE",
                Ordinal::NorthWest => "NW",
                Ordinal::SouthEast => "SE",
                Ordinal::SouthWest => "SW",
            }
        }

        pub fn all() -> Vec<Ordinal> {
            vec![Ordinal::NorthEast, Ordinal::NorthWest, Ordinal::SouthEast, Ordinal::SouthWest]
        }

        pub fn strategy(&self) -> &str {
            match self {
                Ordinal::NorthEast => "Grow + Innovate",
                Ordinal::NorthWest => "Grow + Simplify",
                Ordinal::SouthEast => "Reduce + Innovate",
                Ordinal::SouthWest => "Reduce + Simplify",
            }
        }

        pub fn closest(bearing: &Bearing) -> Ordinal {
            Ordinal::all().into_iter().max_by(|a, b| {
                let da = a.bearing().dot(bearing);
                let db = b.bearing().dot(bearing);
                da.partial_cmp(&db).unwrap()
            }).unwrap()
        }
    }

    /// All 8 compass directions (cardinal + ordinal).
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum CompassDirection {
        C(Cardinal),
        O(Ordinal),
    }

    impl CompassDirection {
        pub fn bearing(&self) -> Bearing {
            match self {
                CompassDirection::C(c) => c.bearing(),
                CompassDirection::O(o) => o.bearing(),
            }
        }

        pub fn all_8() -> Vec<CompassDirection> {
            let mut dirs: Vec<CompassDirection> = Cardinal::all().into_iter().map(CompassDirection::C).collect();
            dirs.extend(Ordinal::all().into_iter().map(CompassDirection::O));
            dirs
        }

        pub fn label(&self) -> &str {
            match self {
                CompassDirection::C(c) => c.label(),
                CompassDirection::O(o) => o.label(),
            }
        }

        pub fn closest(bearing: &Bearing) -> CompassDirection {
            CompassDirection::all_8().into_iter().max_by(|a, b| {
                let da = a.bearing().dot(bearing);
                let db = b.bearing().dot(bearing);
                da.partial_cmp(&db).unwrap()
            }).unwrap()
        }
    }
}

// ============================================================================
// deviation module
// ============================================================================

pub mod deviation {
    use super::bearing::Bearing;

    /// Measure of deviation from an optimal path.
    #[derive(Debug, Clone)]
    pub struct Deviation {
        pub current: Bearing,
        pub optimal: Bearing,
        pub angle: f64,
        pub distance: f64,
    }

    impl Deviation {
        pub fn new(current: &Bearing, optimal: &Bearing) -> Self {
            Self {
                current: current.clone(),
                optimal: optimal.clone(),
                angle: current.angle_to(optimal),
                distance: current.distance_to(optimal),
            }
        }

        pub fn is_on_track(&self, threshold: f64) -> bool {
            self.angle < threshold
        }

        pub fn is_off_track(&self, threshold: f64) -> bool {
            self.angle >= threshold
        }

        /// Correction vector needed to get back on track.
        pub fn correction(&self) -> Bearing {
            let correction_components: Vec<f64> = self.optimal.components.iter()
                .zip(self.current.components.iter())
                .map(|(o, c)| o - c)
                .collect();
            Bearing::new(correction_components)
        }

        /// How much to correct, as a fraction (0.0 = on track, 1.0 = fully off).
        pub fn correction_fraction(&self) -> f64 {
            self.angle / std::f64::consts::PI
        }

        pub fn severity(&self) -> &'static str {
            if self.angle < 0.1 { "minimal" }
            else if self.angle < std::f64::consts::PI / 4.0 { "minor" }
            else if self.angle < std::f64::consts::PI / 2.0 { "moderate" }
            else { "severe" }
        }
    }

    /// Cumulative deviation along a path.
    #[derive(Debug, Clone)]
    pub struct PathDeviation {
        pub deviations: Vec<Deviation>,
    }

    impl PathDeviation {
        pub fn new() -> Self {
            Self { deviations: Vec::new() }
        }

        pub fn add_measurement(&mut self, current: &Bearing, optimal: &Bearing) {
            self.deviations.push(Deviation::new(current, optimal));
        }

        pub fn total_angle(&self) -> f64 {
            self.deviations.iter().map(|d| d.angle).sum()
        }

        pub fn average_angle(&self) -> f64 {
            if self.deviations.is_empty() { return 0.0; }
            self.total_angle() / self.deviations.len() as f64
        }

        pub fn max_deviation(&self) -> Option<&Deviation> {
            self.deviations.iter().max_by(|a, b| a.angle.partial_cmp(&b.angle).unwrap())
        }

        pub fn is_consistently_on_track(&self, threshold: f64) -> bool {
            self.deviations.iter().all(|d| d.is_on_track(threshold))
        }

        pub fn measurement_count(&self) -> usize {
            self.deviations.len()
        }
    }

    impl Default for PathDeviation {
        fn default() -> Self { Self::new() }
    }

    /// Calculate deviation between two bearings.
    pub fn measure(current: &Bearing, optimal: &Bearing) -> Deviation {
        Deviation::new(current, optimal)
    }
}

// ============================================================================
// navigation module
// ============================================================================

pub mod navigation {
    use super::bearing::Bearing;
    use super::deviation::Deviation;

    /// A waypoint in navigation.
    #[derive(Debug, Clone)]
    pub struct Waypoint {
        pub position: Bearing,
        pub label: Option<String>,
        pub reached: bool,
    }

    impl Waypoint {
        pub fn new(bearing: Bearing) -> Self {
            Self { position: bearing, label: None, reached: false }
        }

        pub fn with_label(mut self, label: &str) -> Self {
            self.label = Some(label.to_string());
            self
        }

        pub fn mark_reached(&mut self) {
            self.reached = true;
        }

        pub fn is_reached(&self) -> bool {
            self.reached
        }
    }

    /// A navigation plan through decision space.
    #[derive(Debug, Clone)]
    pub struct NavigationPlan {
        pub waypoints: Vec<Waypoint>,
        pub current_index: usize,
        pub tolerance: f64,
    }

    impl NavigationPlan {
        pub fn new(tolerance: f64) -> Self {
            Self { waypoints: Vec::new(), current_index: 0, tolerance }
        }

        pub fn add_waypoint(&mut self, waypoint: Waypoint) {
            self.waypoints.push(waypoint);
        }

        pub fn current_target(&self) -> Option<&Waypoint> {
            self.waypoints.get(self.current_index)
        }

        pub fn advance(&mut self) -> bool {
            if self.current_index < self.waypoints.len() {
                self.waypoints[self.current_index].mark_reached();
                self.current_index += 1;
                true
            } else {
                false
            }
        }

        pub fn is_complete(&self) -> bool {
            self.current_index >= self.waypoints.len()
        }

        pub fn waypoint_count(&self) -> usize {
            self.waypoints.len()
        }

        pub fn reached_count(&self) -> usize {
            self.waypoints.iter().filter(|w| w.reached).count()
        }

        pub fn progress(&self) -> f64 {
            if self.waypoints.is_empty() { return 1.0; }
            self.reached_count() as f64 / self.waypoints.len() as f64
        }

        /// Update position and check if we've reached the current target.
        pub fn update(&mut self, position: &Bearing) -> Option<Deviation> {
            if let Some(target) = self.current_target() {
                let dev = Deviation::new(position, &target.position);
                if dev.distance < self.tolerance {
                    self.advance();
                }
                Some(dev)
            } else {
                None
            }
        }

        /// Generate a path from start to end with N intermediate waypoints.
        pub fn interpolate_path(start: &Bearing, end: &Bearing, steps: usize, tolerance: f64) -> NavigationPlan {
            let mut plan = NavigationPlan::new(tolerance);
            for i in 1..=steps {
                let t = i as f64 / steps as f64;
                let bearing = super::bearing::interpolate(start, end, t);
                plan.add_waypoint(Waypoint::new(bearing));
            }
            plan
        }

        /// Compute total path length.
        pub fn total_path_length(&self) -> f64 {
            self.waypoints.windows(2)
                .map(|w| w[0].position.distance_to(&w[1].position))
                .sum()
        }
    }

    /// Navigate from a start point toward a goal, avoiding obstacles.
    pub fn navigate_around_obstacle(
        start: &Bearing,
        goal: &Bearing,
        obstacle: &Bearing,
        obstacle_radius: f64,
        step_size: f64,
    ) -> Vec<Bearing> {
        let mut path = vec![start.clone()];
        let mut current = start.clone();
        let max_steps = 100;

        for _ in 0..max_steps {
            let to_goal = Bearing::new(
                goal.components.iter().zip(current.components.iter())
                    .map(|(g, c)| g - c)
                    .collect()
            );

            if to_goal.magnitude < step_size {
                path.push(goal.clone());
                break;
            }

            let dist_to_obstacle = current.distance_to(obstacle);
            let mut direction = to_goal.normalize();

            if dist_to_obstacle < obstacle_radius {
                // Steer away from obstacle
                let away = Bearing::new(
                    current.components.iter().zip(obstacle.components.iter())
                        .map(|(c, o)| c - o)
                        .collect()
                );
                direction = direction.add(&away.normalize()).normalize();
            }

            current = current.add(&direction.scale(step_size));
            path.push(current.clone());
        }

        path
    }
}

// Re-exports
pub use bearing::{Bearing, axis_bearing, interpolate};
pub use cardinal::{Cardinal, score_cardinals, dominant};
pub use ordinal::{Ordinal, CompassDirection};
pub use deviation::{Deviation, PathDeviation, measure};
pub use navigation::{Waypoint, NavigationPlan, navigate_around_obstacle};

#[cfg(test)]
mod tests {
    use super::*;

    // ---- bearing tests (15) ----

    #[test]
    fn test_bearing_new() {
        let b = bearing::Bearing::new(vec![3.0, 4.0]);
        assert!((b.magnitude - 5.0).abs() < 0.01);
        assert_eq!(b.dimensions(), 2);
    }

    #[test]
    fn test_bearing_zero() {
        let b = bearing::Bearing::new(vec![0.0, 0.0, 0.0]);
        assert!(b.is_zero());
    }

    #[test]
    fn test_bearing_normalize() {
        let b = bearing::Bearing::new(vec![3.0, 4.0]).normalize();
        assert!((b.magnitude - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_bearing_dot() {
        let a = bearing::Bearing::new(vec![1.0, 0.0]);
        let b = bearing::Bearing::new(vec![0.0, 1.0]);
        assert!((a.dot(&b)).abs() < 0.01);
    }

    #[test]
    fn test_bearing_angle_parallel() {
        let a = bearing::Bearing::new(vec![1.0, 0.0]);
        let b = bearing::Bearing::new(vec![2.0, 0.0]);
        assert!(a.angle_to(&b) < 0.01);
    }

    #[test]
    fn test_bearing_angle_perpendicular() {
        let a = bearing::Bearing::new(vec![1.0, 0.0]);
        let b = bearing::Bearing::new(vec![0.0, 1.0]);
        assert!((a.angle_to(&b) - std::f64::consts::PI / 2.0).abs() < 0.01);
    }

    #[test]
    fn test_bearing_add() {
        let a = bearing::Bearing::new(vec![1.0, 2.0]);
        let b = bearing::Bearing::new(vec![3.0, 4.0]);
        let sum = a.add(&b);
        assert_eq!(sum.components, vec![4.0, 6.0]);
    }

    #[test]
    fn test_bearing_scale() {
        let b = bearing::Bearing::new(vec![1.0, 2.0]).scale(3.0);
        assert_eq!(b.components, vec![3.0, 6.0]);
    }

    #[test]
    fn test_bearing_distance() {
        let a = bearing::Bearing::new(vec![0.0, 0.0]);
        let b = bearing::Bearing::new(vec![3.0, 4.0]);
        assert!((a.distance_to(&b) - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_bearing_project() {
        let a = bearing::Bearing::new(vec![3.0, 4.0]);
        let b = bearing::Bearing::new(vec![1.0, 0.0]);
        let proj = a.project_onto(&b);
        assert!((proj.components[0] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_bearing_reflect() {
        let v = bearing::Bearing::new(vec![1.0, -1.0]);
        let normal = bearing::Bearing::new(vec![0.0, 1.0]);
        let reflected = v.reflect(&normal);
        assert!((reflected.components[0] - 1.0).abs() < 0.01);
        assert!((reflected.components[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_axis_bearing() {
        let b = bearing::axis_bearing(1, 3);
        assert_eq!(b.components, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_interpolate() {
        let a = bearing::Bearing::new(vec![0.0, 0.0]);
        let b = bearing::Bearing::new(vec![10.0, 10.0]);
        let mid = bearing::interpolate(&a, &b, 0.5);
        assert!((mid.components[0] - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_bearing_component() {
        let b = bearing::Bearing::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(b.component(1), Some(2.0));
        assert_eq!(b.component(5), None);
    }

    #[test]
    fn test_bearing_label() {
        let b = bearing::Bearing::new(vec![1.0]).with_label("test");
        assert_eq!(b.label, Some("test".to_string()));
    }

    // ---- cardinal tests (10) ----

    #[test]
    fn test_cardinal_bearing() {
        let n = cardinal::Cardinal::North.bearing();
        assert_eq!(n.components, vec![0.0, 1.0]);
    }

    #[test]
    fn test_cardinal_opposite() {
        assert_eq!(cardinal::Cardinal::North.opposite(), cardinal::Cardinal::South);
        assert_eq!(cardinal::Cardinal::East.opposite(), cardinal::Cardinal::West);
    }

    #[test]
    fn test_cardinal_label() {
        assert_eq!(cardinal::Cardinal::North.label(), "North");
    }

    #[test]
    fn test_cardinal_all() {
        assert_eq!(cardinal::Cardinal::all().len(), 4);
    }

    #[test]
    fn test_cardinal_closest() {
        let b = bearing::Bearing::new(vec![0.0, 1.0]);
        assert_eq!(cardinal::Cardinal::closest(&b), cardinal::Cardinal::North);
    }

    #[test]
    fn test_cardinal_strategy() {
        let s = cardinal::Cardinal::North.strategy();
        assert_eq!(s, "Expand/Grow");
    }

    #[test]
    fn test_score_cardinals() {
        let context = bearing::Bearing::new(vec![1.0, 1.0]);
        let scores = cardinal::score_cardinals(&context);
        assert_eq!(scores.len(), 4);
    }

    #[test]
    fn test_dominant() {
        let context = bearing::Bearing::new(vec![5.0, 0.0]);
        assert_eq!(cardinal::dominant(&context), cardinal::Cardinal::East);
    }

    #[test]
    fn test_cardinal_bearings_unit() {
        for c in cardinal::Cardinal::all() {
            assert!((c.bearing().magnitude - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_cardinal_closest_south() {
        let b = bearing::Bearing::new(vec![0.0, -10.0]);
        assert_eq!(cardinal::Cardinal::closest(&b), cardinal::Cardinal::South);
    }

    // ---- ordinal tests (10) ----

    #[test]
    fn test_ordinal_bearing() {
        let ne = ordinal::Ordinal::NorthEast.bearing();
        assert!((ne.magnitude - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_ordinal_cardinals() {
        let (a, b) = ordinal::Ordinal::NorthEast.cardinals();
        assert_eq!(a, cardinal::Cardinal::North);
        assert_eq!(b, cardinal::Cardinal::East);
    }

    #[test]
    fn test_ordinal_label() {
        assert_eq!(ordinal::Ordinal::SouthWest.label(), "SW");
    }

    #[test]
    fn test_ordinal_all() {
        assert_eq!(ordinal::Ordinal::all().len(), 4);
    }

    #[test]
    fn test_ordinal_closest() {
        let b = bearing::Bearing::new(vec![1.0, 1.0]);
        assert_eq!(ordinal::Ordinal::closest(&b), ordinal::Ordinal::NorthEast);
    }

    #[test]
    fn test_ordinal_strategy() {
        assert_eq!(ordinal::Ordinal::NorthEast.strategy(), "Grow + Innovate");
    }

    #[test]
    fn test_compass_direction_8() {
        assert_eq!(ordinal::CompassDirection::all_8().len(), 8);
    }

    #[test]
    fn test_compass_closest() {
        let b = bearing::Bearing::new(vec![-1.0, 1.0]);
        let closest = ordinal::CompassDirection::closest(&b);
        assert_eq!(closest, ordinal::CompassDirection::O(ordinal::Ordinal::NorthWest));
    }

    #[test]
    fn test_compass_label() {
        let dir = ordinal::CompassDirection::C(cardinal::Cardinal::North);
        assert_eq!(dir.label(), "North");
    }

    #[test]
    fn test_compass_bearing() {
        for dir in ordinal::CompassDirection::all_8() {
            assert!((dir.bearing().magnitude - 1.0).abs() < 0.01 || dir.bearing().magnitude > 0.0);
        }
    }

    // ---- deviation tests (10) ----

    #[test]
    fn test_deviation_new() {
        let a = bearing::Bearing::new(vec![1.0, 0.0]);
        let b = bearing::Bearing::new(vec![0.0, 1.0]);
        let dev = deviation::Deviation::new(&a, &b);
        assert!((dev.angle - std::f64::consts::PI / 2.0).abs() < 0.01);
    }

    #[test]
    fn test_deviation_on_track() {
        let a = bearing::Bearing::new(vec![1.0, 0.0]);
        let dev = deviation::Deviation::new(&a, &a);
        assert!(dev.is_on_track(0.1));
    }

    #[test]
    fn test_deviation_off_track() {
        let a = bearing::Bearing::new(vec![1.0, 0.0]);
        let b = bearing::Bearing::new(vec![0.0, 1.0]);
        let dev = deviation::Deviation::new(&a, &b);
        assert!(dev.is_off_track(0.1));
    }

    #[test]
    fn test_deviation_correction() {
        let current = bearing::Bearing::new(vec![1.0, 0.0]);
        let optimal = bearing::Bearing::new(vec![0.0, 1.0]);
        let dev = deviation::Deviation::new(&current, &optimal);
        let correction = dev.correction();
        assert!((correction.components[0] - (-1.0)).abs() < 0.01);
        assert!((correction.components[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_deviation_severity() {
        let a = bearing::Bearing::new(vec![1.0, 0.0]);
        let dev = deviation::Deviation::new(&a, &a);
        assert_eq!(dev.severity(), "minimal");
    }

    #[test]
    fn test_path_deviation() {
        let mut pd = deviation::PathDeviation::new();
        let a = bearing::Bearing::new(vec![1.0, 0.0]);
        let b = bearing::Bearing::new(vec![0.0, 1.0]);
        pd.add_measurement(&a, &b);
        assert_eq!(pd.measurement_count(), 1);
    }

    #[test]
    fn test_path_deviation_total_angle() {
        let mut pd = deviation::PathDeviation::new();
        let a = bearing::Bearing::new(vec![1.0, 0.0]);
        let b = bearing::Bearing::new(vec![0.0, 1.0]);
        pd.add_measurement(&a, &b);
        pd.add_measurement(&a, &a);
        assert!(pd.total_angle() > 0.0);
    }

    #[test]
    fn test_path_deviation_consistently_on_track() {
        let mut pd = deviation::PathDeviation::new();
        let a = bearing::Bearing::new(vec![1.0, 0.0]);
        pd.add_measurement(&a, &a);
        pd.add_measurement(&a, &a);
        assert!(pd.is_consistently_on_track(0.1));
    }

    #[test]
    fn test_measure_function() {
        let a = bearing::Bearing::new(vec![1.0, 0.0]);
        let b = bearing::Bearing::new(vec![0.0, 1.0]);
        let dev = deviation::measure(&a, &b);
        assert!(dev.angle > 0.0);
    }

    #[test]
    fn test_correction_fraction() {
        let a = bearing::Bearing::new(vec![1.0, 0.0]);
        let b = bearing::Bearing::new(vec![0.0, 1.0]);
        let dev = deviation::Deviation::new(&a, &b);
        assert!((dev.correction_fraction() - 0.5).abs() < 0.01);
    }

    // ---- navigation tests (10) ----

    #[test]
    fn test_waypoint_new() {
        let wp = navigation::Waypoint::new(bearing::Bearing::new(vec![1.0, 0.0]));
        assert!(!wp.is_reached());
        assert!(wp.label.is_none());
    }

    #[test]
    fn test_waypoint_reached() {
        let mut wp = navigation::Waypoint::new(bearing::Bearing::new(vec![1.0, 0.0]));
        wp.mark_reached();
        assert!(wp.is_reached());
    }

    #[test]
    fn test_navigation_plan_new() {
        let plan = navigation::NavigationPlan::new(0.1);
        assert!(plan.is_complete());
        assert_eq!(plan.waypoint_count(), 0);
    }

    #[test]
    fn test_navigation_plan_waypoints() {
        let mut plan = navigation::NavigationPlan::new(0.1);
        plan.add_waypoint(navigation::Waypoint::new(bearing::Bearing::new(vec![1.0])));
        plan.add_waypoint(navigation::Waypoint::new(bearing::Bearing::new(vec![2.0])));
        assert_eq!(plan.waypoint_count(), 2);
        assert!(!plan.is_complete());
    }

    #[test]
    fn test_navigation_plan_advance() {
        let mut plan = navigation::NavigationPlan::new(0.1);
        plan.add_waypoint(navigation::Waypoint::new(bearing::Bearing::new(vec![1.0])));
        plan.advance();
        assert!(plan.is_complete());
        assert_eq!(plan.reached_count(), 1);
    }

    #[test]
    fn test_navigation_plan_progress() {
        let mut plan = navigation::NavigationPlan::new(0.1);
        plan.add_waypoint(navigation::Waypoint::new(bearing::Bearing::new(vec![1.0])));
        plan.add_waypoint(navigation::Waypoint::new(bearing::Bearing::new(vec![2.0])));
        assert!((plan.progress()).abs() < 0.01);
        plan.advance();
        assert!((plan.progress() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_interpolate_path() {
        let start = bearing::Bearing::new(vec![0.0]);
        let end = bearing::Bearing::new(vec![10.0]);
        let plan = navigation::NavigationPlan::interpolate_path(&start, &end, 5, 0.1);
        assert_eq!(plan.waypoint_count(), 5);
    }

    #[test]
    fn test_navigation_update() {
        let mut plan = navigation::NavigationPlan::new(2.0);
        plan.add_waypoint(navigation::Waypoint::new(bearing::Bearing::new(vec![1.0])));
        let pos = bearing::Bearing::new(vec![0.5]);
        let dev = plan.update(&pos);
        assert!(dev.is_some());
    }

    #[test]
    fn test_navigate_around_obstacle() {
        let start = bearing::Bearing::new(vec![0.0, 0.0]);
        let goal = bearing::Bearing::new(vec![10.0, 0.0]);
        let obstacle = bearing::Bearing::new(vec![5.0, 0.0]);
        let path = navigation::navigate_around_obstacle(&start, &goal, &obstacle, 1.5, 1.0);
        assert!(path.len() > 2);
    }

    #[test]
    fn test_total_path_length() {
        let mut plan = navigation::NavigationPlan::new(0.1);
        plan.add_waypoint(navigation::Waypoint::new(bearing::Bearing::new(vec![0.0])));
        plan.add_waypoint(navigation::Waypoint::new(bearing::Bearing::new(vec![5.0])));
        plan.add_waypoint(navigation::Waypoint::new(bearing::Bearing::new(vec![10.0])));
        assert!((plan.total_path_length() - 10.0).abs() < 0.01);
    }
}
