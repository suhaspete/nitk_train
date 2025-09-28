

import pygame
import sys
import numpy as np
from datetime import datetime, timedelta
import random
import math

class Location:
    """Represents a location on the railway network"""
    def __init__(self, name, location_type, position_km, side_tracks=0):
        self.name = name
        self.type = location_type  # 'city', 'town', 'village', 'open'
        self.position_km = position_km
        self.side_tracks = side_tracks
        self.has_double_track = location_type in ['city', 'town', 'village']
        self.occupied_side_tracks = 0  # Track side track usage

class RailwayTrack:
    """Railway track with single/double sections and side tracks"""
    def __init__(self, name="main_line", total_length_km=400):
        self.name = name
        self.total_length_km = total_length_km
        self.locations = self._create_locations()
        self.segments = self._create_segments()
        
    def _create_locations(self):
        """Create realistic railway locations"""
        return [
            Location("Mumbai", "city", 0, side_tracks=4),
            Location("Thane", "city", 25, side_tracks=3),
            Location("Kalyan", "town", 55, side_tracks=2),
            Location("Karjat", "village", 85, side_tracks=2),
            Location("Lonavala", "village", 115, side_tracks=2),
            Location("Pune", "city", 150, side_tracks=4),
            Location("Satara", "town", 200, side_tracks=2),
            Location("Kolhapur", "village", 250, side_tracks=1),
            Location("Belgaum", "city", 300, side_tracks=3),
            Location("Bangalore", "city", 400, side_tracks=4)
        ]
    
    def _create_segments(self):
        """Create track segments between locations"""
        segments = []
        for i in range(len(self.locations) - 1):
            start_loc = self.locations[i]
            end_loc = self.locations[i + 1]
            
            # Determine track type based on locations
            has_double_track = start_loc.has_double_track or end_loc.has_double_track
            
            segments.append({
                'start': start_loc,
                'end': end_loc,
                'start_km': start_loc.position_km,
                'end_km': end_loc.position_km,
                'length_km': end_loc.position_km - start_loc.position_km,
                'has_double_track': has_double_track,
                'capacity': 2 if has_double_track else 1,
                'trains': []
            })
        return segments
    
    def get_segment_at_position(self, position_km):
        """Get track segment at given position"""
        for segment in self.segments:
            if segment['start_km'] <= position_km <= segment['end_km']:
                return segment
        return None
    
    def is_in_double_track_area(self, position_km):
        """Check if position is in double-track area (village/city)"""
        for location in self.locations:
            # Check if within 10km of a settlement (double-track area)
            if abs(location.position_km - position_km) <= 10 and location.has_double_track:
                return True, location
        return False, None
    
    def find_nearest_side_track(self, position_km):
        """Find nearest location with available side tracks"""
        candidates = []
        for location in self.locations:
            if location.side_tracks > location.occupied_side_tracks:
                distance = abs(location.position_km - position_km)
                candidates.append((distance, location))
        
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        return None
    
    def find_next_village_ahead(self, position_km):
        """Find the next village/city AHEAD of current position with available side tracks"""
        candidates = []
        for location in self.locations:
            # Only consider locations AHEAD (greater position) and with available side tracks
            if (location.position_km > position_km and 
                location.has_double_track and 
                location.side_tracks > location.occupied_side_tracks):
                distance = location.position_km - position_km
                candidates.append((distance, location))
        
        if candidates:
            candidates.sort(key=lambda x: x[0])  # Sort by distance, closest first
            return candidates[0][1]
        return None
    
    def is_in_single_track_section(self, position_km):
        """Check if position is in single-track section (outside villages/cities)"""
        is_double_track, _ = self.is_in_double_track_area(position_km)
        return not is_double_track
    
    def get_next_double_track_location(self, position_km):
        """Get the next double-track location ahead"""
        for location in self.locations:
            if location.position_km > position_km and location.has_double_track:
                return location
        return None

class Train:
    """Train with realistic speeds and behavior"""
    def __init__(self, id, name, priority, base_speed, scheduled_start):
        self.id = id
        self.name = name
        self.priority = priority  # 1=highest, 4=lowest
        self.base_speed = base_speed  # km/h
        self.current_speed = base_speed
        self.scheduled_start = scheduled_start
        self.position_km = 0.0
        self.delay_minutes = 0
        self.is_stopped = False
        self.stop_reason = ""
        self.has_started = False
        self.destination_reached = False
        self.is_slowing_for_delayed_train = False
        self.emergency_stopped = False  # NEW: Track emergency stops
        
        # Side track management
        self.is_on_side_track = False
        self.side_track_location = None
        self.waiting_for_train = None
        self.side_track_timer = 0
        
        # Speed matching for overtaking
        self.original_speed = base_speed
        self.is_speed_matched = False
        self.speed_matched_to_train = None
        self.waiting_for_double_track = False
        
        # NEW: Phase tracking
        self.current_phase = "NORMAL"  # NORMAL, MONITOR, PROGRESSIVE, SPEED_MATCH, EMERGENCY
        self.phase_target_train = None
        
        # Visual properties
        self.color = self._get_priority_color(priority)
        self.animation_offset = 0
        self.last_update_time = None
        
        # State tracking
        self.total_delay_accumulated = 0
        self.times_rerouted = 0
        
    def _get_priority_color(self, priority):
        """Get color based on train priority"""
        colors = {
            1: (255, 0, 0),    # Red - Highest priority
            2: (255, 140, 0),  # Dark orange - High priority
            3: (0, 100, 255),  # Blue - Medium priority
            4: (100, 100, 100) # Gray - Lowest priority
        }
        return colors.get(priority, (0, 0, 0))
    
    def can_start(self, current_time):
        """Check if train can start based on schedule"""
        return current_time >= self.scheduled_start and not self.has_started
    
    def start_journey(self, current_time):
        """Start the train's journey"""
        if self.can_start(current_time):
            self.has_started = True
            self.last_update_time = current_time
            self.is_stopped = False
            return True
        return False
    
    def update_position(self, time_delta_minutes, track):
        """Update train position with corrected delay handling"""
        if not self.has_started or self.destination_reached:
            return
        
        # Trains on yellow tracks cannot move at all
        if self.is_on_side_track:
            self.is_stopped = True
            if self.side_track_location:
                self.stop_reason = f"Stationary on yellow track at {self.side_track_location.name}"
            return
        
        # Handle delays
        if self.delay_minutes > 0:
            self.delay_minutes -= time_delta_minutes
            self.is_stopped = True
            self.stop_reason = f"Delayed: {self.delay_minutes:.1f}min remaining"
            if self.delay_minutes <= 0:
                self.delay_minutes = 0
                self.is_stopped = False
                self.stop_reason = ""
                self.emergency_stopped = False  # Clear emergency stop when delay ends
            return
        
        # Don't move if stopped for other reasons (including emergency stop)
        if self.is_stopped or self.emergency_stopped:
            return
        
        # Calculate movement
        speed_km_per_min = self.current_speed / 60  # Convert to km/min
        distance_increment = speed_km_per_min * time_delta_minutes
        
        # Move train
        new_position = min(track.total_length_km, self.position_km + distance_increment)
        self.position_km = new_position
        
        # Check if destination reached
        if self.position_km >= track.total_length_km:
            self.destination_reached = True
            self.is_stopped = True
            self.stop_reason = "Journey completed"
    
    def add_delay(self, minutes, reason):
        """Add delay to train"""
        self.delay_minutes += minutes
        self.total_delay_accumulated += minutes
        self.is_stopped = True
        self.stop_reason = f"Delayed: {reason} ({self.delay_minutes:.1f}min)"
    
    def emergency_stop(self, delayed_train, reason):
        """NEW: Emergency stop the train completely"""
        print(f"EMERGENCY STOP: {self.name} - {reason}")
        self.emergency_stopped = True
        self.current_speed = 0
        self.is_stopped = True
        self.current_phase = "EMERGENCY"
        self.phase_target_train = delayed_train
        self.stop_reason = f"EMERGENCY: {reason}"
        # Inherit the SAME delay as the delayed train
        self.add_delay(delayed_train.delay_minutes, f"Inherited from {delayed_train.name}")
    
    def set_phase(self, phase, distance, delayed_train):
        """NEW: Set the current phase and adjust speed accordingly"""
        old_phase = self.current_phase
        self.current_phase = phase
        self.phase_target_train = delayed_train
        
        if phase != old_phase:
            print(f"PHASE CHANGE: {self.name} {old_phase} → {phase} (Distance: {distance:.1f}km)")
        
        if phase == "MONITOR":
            # Phase 1: Far away, maintain original speed
            self.current_speed = self.original_speed
            self.stop_reason = f"Phase 1: Monitoring {delayed_train.name}"
            
        elif phase == "PROGRESSIVE":
            # Phase 2: Progressive slowdown from 35km to 20km
            # Linear interpolation between original speed and delayed train speed
            distance_factor = (distance - 20) / (35 - 20)  # 1.0 at 35km, 0.0 at 20km
            distance_factor = max(0, min(1, distance_factor))
            
            target_speed = delayed_train.current_speed + (self.original_speed - delayed_train.current_speed) * distance_factor
            self.current_speed = max(delayed_train.current_speed, target_speed)
            self.stop_reason = f"Phase 2: Progressive slowdown ({self.current_speed:.0f}km/h)"
            
        elif phase == "SPEED_MATCH":
            # Phase 3: Speed matching
            self.current_speed = delayed_train.current_speed
            self.is_speed_matched = True
            self.speed_matched_to_train = delayed_train
            self.stop_reason = f"Phase 3: Speed matching {delayed_train.name}"
            
        elif phase == "EMERGENCY":
            # Phase 4: Emergency stop
            self.current_speed = 0
            self.emergency_stopped = True
            self.is_stopped = True
            self.stop_reason = f"Phase 4: Emergency stop - collision imminent"
    
    def resume_normal_speed(self):
        """Enhanced resume normal speed with complete state reset"""
        if self.current_phase != "NORMAL" or self.emergency_stopped or self.is_speed_matched:
            print(f"FULL RECOVERY: {self.name} → {self.original_speed}km/h (was {self.current_speed}km/h)")
            
            # Reset all speed and phase states
            self.current_speed = self.original_speed
            self.current_phase = "NORMAL"
            self.phase_target_train = None
            self.is_slowing_for_delayed_train = False
            self.is_speed_matched = False
            self.speed_matched_to_train = None
            self.emergency_stopped = False
            self.is_stopped = False
            self.stop_reason = ""
    
    def move_to_side_track(self, side_track_location, reason):
        """Move train to side track"""
        if side_track_location and side_track_location.occupied_side_tracks < side_track_location.side_tracks:
            self.is_on_side_track = True
            self.side_track_location = side_track_location
            self.position_km = side_track_location.position_km
            self.is_stopped = True
            self.stop_reason = f"Yellow track at {side_track_location.name}: {reason}"
            self.times_rerouted += 1
            side_track_location.occupied_side_tracks += 1
            print(f"YELLOW TRACK: {self.name} moved to {side_track_location.name} - {reason}")
            return True
        return False
    
    def return_to_main_track(self):
        """Return train from side track to main track"""
        if self.is_on_side_track and self.side_track_location:
            self.side_track_location.occupied_side_tracks -= 1
            self.is_on_side_track = False
            old_location = self.side_track_location.name
            self.side_track_location = None
            self.waiting_for_train = None
            self.is_stopped = False
            self.stop_reason = ""
            print(f"MAIN TRACK: {self.name} returned from {old_location}")

class DynamicRailwayScheduler:
    """Railway scheduler with 4-phase progressive delay handling"""
    
    def __init__(self):
        self.trains = []
        self.track = RailwayTrack()
        self.current_time = datetime.now().replace(hour=6, minute=0, second=0, microsecond=0)
        self.simulation_minutes = 0
        self.delay_events = []
        self.overtaking_events = []
        
    def create_dynamic_schedule(self, train_configs):
        """Create schedule with dynamic starting order"""
        shuffled_configs = train_configs.copy()
        random.shuffle(shuffled_configs)
        
        print("=== DYNAMIC TRAIN SCHEDULING ===")
        print("Trains will start in mixed priority order with 1-hour intervals:")
        
        start_time = self.current_time
        for i, config in enumerate(shuffled_configs):
            scheduled_time = start_time + timedelta(hours=i)
            
            train = Train(
                config['id'],
                config['name'],
                config['priority'],
                config['speed'],
                scheduled_time
            )
            
            self.trains.append(train)
            print(f"  {train.name} (P{train.priority}, {train.base_speed}km/h) -> {scheduled_time.strftime('%H:%M')}")
        
        print(f"Total trains scheduled: {len(self.trains)}")
        return self.trains
    
    def simulate_step(self, time_delta_minutes=0.05):
        """Simulate one time step with 4-phase delay logic"""
        self.current_time += timedelta(minutes=time_delta_minutes)
        self.simulation_minutes += time_delta_minutes
        
        # Start ready trains
        for train in self.trains:
            if train.can_start(self.current_time):
                train.start_journey(self.current_time)
                print(f"\n🚀 TRAIN STARTED: {train.name} (P{train.priority}, {train.original_speed}km/h) at {self.current_time.strftime('%H:%M')}")
        
        # Handle 4-phase delay consequences
        self._handle_four_phase_delay_logic()
        
        # Handle delay recovery
        self._handle_delay_recovery()
        
        # Handle overtaking logic
        self._handle_overtaking_logic()
        
        # Update all train positions
        for train in self.trains:
            if train.has_started and not train.destination_reached:
                train.update_position(time_delta_minutes, self.track)
        
        # Process side track returns
        self._process_side_track_returns()
        
        # Update track occupancy
        self._update_track_occupancy()
    
    def trigger_user_delay(self):
        """Trigger delay with 4-phase logic"""
        active_trains = [t for t in self.trains if t.has_started and not t.destination_reached and not t.is_on_side_track]
        if not active_trains:
            return None
        
        # Select random train for delay
        delayed_train = random.choice(active_trains)
        delay_amount = random.randint(15, 45)  # 15-45 minute delay
        
        print(f"\n=== USER DELAY TRIGGERED ===")
        print(f"Train: {delayed_train.name} (Priority {delayed_train.priority})")
        print(f"Position: {delayed_train.position_km:.1f}km")
        print(f"Delay: {delay_amount} minutes")
        
        # Add the delay
        delayed_train.add_delay(delay_amount, "User-triggered delay")
        
        return delayed_train
    
    def _handle_four_phase_delay_logic(self):
        """NEW: Implement universal 4-phase progressive delay handling with chain reactions"""
        # Find ALL trains that are currently delayed (including chain delays)
        delayed_trains = [t for t in self.trains if t.delay_minutes > 0 and t.has_started]
        
        for delayed_train in delayed_trains:
            # Find ALL trains near the delayed train within 50km (regardless of position)
            approaching_trains = []
            for t in self.trains:
                if (t != delayed_train and 
                    t.has_started and 
                    not t.destination_reached and
                    not t.is_on_side_track):
                    
                    distance = abs(delayed_train.position_km - t.position_km)
                    if distance <= 50 and distance > 0:  # Within 50km monitoring range
                        approaching_trains.append((distance, t))
            
            # Sort by distance (closest first)
            approaching_trains.sort(key=lambda x: x[0])
            
            # Apply universal 4-phase logic to each approaching train
            for distance, approaching_train in approaching_trains:
                self._apply_four_phase_logic(delayed_train, approaching_train, distance)
        
        # NEW: Handle chain reaction delays
        self._handle_chain_reaction_delays()
    
    def _apply_four_phase_logic(self, delayed_train, approaching_train, distance):
        """NEW: Apply the 4-phase progressive delay logic UNIVERSALLY (regardless of speed)"""
        
        # Apply 4-phase logic to ANY train approaching a delayed train
        # Remove speed condition - works for faster, slower, or equal speed trains
        
        # Determine phase based on distance
        if distance > 35:
            # Phase 1: Far Away - Monitor Only (50-35km)
            new_phase = "MONITOR"
        elif distance > 20:
            # Phase 2: Getting Closer - Progressive Slowdown (35-20km)
            new_phase = "PROGRESSIVE"
        elif distance > 10:
            # Phase 3: Very Close - Speed Matching (20-10km)
            new_phase = "SPEED_MATCH"
        else:
            # Phase 4: Emergency Zone - Emergency Stop (≤10km)
            new_phase = "EMERGENCY"
        
        # Apply phase if it's different from current phase
        if approaching_train.current_phase != new_phase or approaching_train.phase_target_train != delayed_train:
            approaching_train.set_phase(new_phase, distance, delayed_train)
        
        # Special handling for Phase 4 - Emergency Stop (UNIVERSAL)
        if new_phase == "EMERGENCY" and not approaching_train.emergency_stopped:
            # Calculate collision prediction
            speed_diff = approaching_train.current_speed - delayed_train.current_speed
            if speed_diff > 0:
                time_to_collision = distance / (speed_diff / 60)  # minutes
            else:
                time_to_collision = float('inf')
            
            print(f"\nUNIVERSAL EMERGENCY ANALYSIS:")
            print(f"  Approaching train: {approaching_train.name} ({approaching_train.original_speed}km/h)")
            print(f"  Delayed train: {delayed_train.name} ({delayed_train.original_speed}km/h)")
            print(f"  Distance: {distance:.1f}km")
            print(f"  Speed difference: {speed_diff:.1f}km/h")
            print(f"  Time to collision: {time_to_collision:.1f} minutes")
            print(f"  Delay remaining: {delayed_train.delay_minutes:.1f} minutes")
            
            # Emergency stop with delay inheritance (REGARDLESS of speed relationship)
            if time_to_collision <= delayed_train.delay_minutes:
                print(f"  → COLLISION INEVITABLE - Emergency stop + delay inheritance")
                approaching_train.emergency_stop(delayed_train, f"Collision inevitable with {delayed_train.name}")
            else:
                print(f"  → Delay will end in time - maintaining emergency protocols")
                # Still emergency stop but with possibility of recovery
                approaching_train.current_speed = 0
                approaching_train.emergency_stopped = True
                approaching_train.is_stopped = True
                approaching_train.stop_reason = f"Emergency hold - {delayed_train.name} delayed"
    
    def _handle_chain_reaction_delays(self):
        """NEW: Handle chain reaction delays - when delayed trains cause other delays"""
        # Find trains that are delayed due to other trains
        secondary_delayed_trains = [t for t in self.trains 
                                  if (t.has_started and 
                                      not t.destination_reached and
                                      t.delay_minutes > 0 and
                                      t.phase_target_train is not None)]
        
        for secondary_train in secondary_delayed_trains:
            # This train is delayed due to another train - now check trains behind IT
            trains_behind_secondary = []
            for t in self.trains:
                if (t != secondary_train and
                    t.has_started and 
                    not t.destination_reached and
                    not t.is_on_side_track and
                    t != secondary_train.phase_target_train):  # Don't include the original delayed train
                    
                    distance = abs(secondary_train.position_km - t.position_km)
                    if distance <= 50 and distance > 0:
                        trains_behind_secondary.append((distance, t))
            
            # Sort by distance
            trains_behind_secondary.sort(key=lambda x: x[0])
            
            # Apply 4-phase logic for the secondary delayed train
            for distance, train_behind in trains_behind_secondary:
                # Treat the secondary delayed train as a delayed train for these followers
                print(f"\nCHAIN REACTION: {train_behind.name} approaching secondary delayed {secondary_train.name}")
                self._apply_four_phase_logic(secondary_train, train_behind, distance)
    
    def _handle_delay_recovery(self):
        """Enhanced delay recovery handling with proper speed restoration"""
        for train in self.trains:
            # Check if train is in any delay-related phase but the target train is no longer delayed
            if (train.current_phase != "NORMAL" and 
                train.phase_target_train and 
                train.phase_target_train.delay_minutes <= 0):
                
                delayed_train = train.phase_target_train
                distance = abs(delayed_train.position_km - train.position_km)
                
                print(f"\nDELAY RECOVERY: {train.name} can potentially resume - {delayed_train.name} no longer delayed")
                print(f"  Current distance: {distance:.1f}km")
                print(f"  Current phase: {train.current_phase}")
                print(f"  Current speed: {train.current_speed}km/h, Original speed: {train.original_speed}km/h")
                
                # Recovery logic based on distance
                if distance > 35:  # Far enough for complete recovery
                    print(f"  → FULL RECOVERY - Restoring to {train.original_speed}km/h")
                    train.resume_normal_speed()
                elif distance > 20:  # Partial recovery
                    print(f"  → PARTIAL RECOVERY - Gradual speed increase")
                    train.set_phase("PROGRESSIVE", distance, delayed_train)
                elif distance > 10:  # Still close, speed matching
                    print(f"  → CAUTIOUS RECOVERY - Maintaining close monitoring")
                    train.set_phase("SPEED_MATCH", distance, delayed_train)
                else:
                    print(f"  → TOO CLOSE - Maintaining emergency protocols")
                    # Keep emergency status until more distance
            
            # NEW: Handle trains that had their own delay (not due to other trains) and should recover
            elif (train.delay_minutes <= 0 and 
                  train.emergency_stopped and 
                  train.phase_target_train is None):
                print(f"\nDIRECT DELAY RECOVERY: {train.name} own delay ended - resuming normal speed")
                train.resume_normal_speed()
    
    def _handle_overtaking_logic(self):
        """Handle normal overtaking when no delays are involved"""
        active_trains = [t for t in self.trains 
                        if (t.has_started and 
                            not t.destination_reached and
                            not t.is_stopped and
                            t.delay_minutes <= 0 and
                            t.current_phase == "NORMAL")]
        
        if len(active_trains) < 2:
            return
            
        # Check every pair of trains for overtaking opportunities
        for i in range(len(active_trains)):
            for j in range(len(active_trains)):
                if i == j:
                    continue
                    
                train_behind = active_trains[i]
                train_ahead = active_trains[j]
                
                # Only process if conditions are met for overtaking
                if (train_behind.position_km < train_ahead.position_km and
                    train_behind.original_speed > train_ahead.original_speed and
                    not train_behind.is_on_side_track and
                    not train_ahead.is_on_side_track):
                    
                    distance_gap = train_ahead.position_km - train_behind.position_km
                    
                    # When faster train gets within 25km
                    if distance_gap <= 25 and distance_gap > 0:
                        # Step 1: Slow down faster train
                        if not train_behind.is_speed_matched:
                            print(f"\n🚂 OVERTAKING STEP 1: {train_behind.name} slowing to match {train_ahead.name}")
                            train_behind.current_speed = train_ahead.current_speed
                            train_behind.is_speed_matched = True
                            train_behind.speed_matched_to_train = train_ahead
                        
                        # Step 2: Move slower train to side track
                        if not train_ahead.is_on_side_track:
                            side_track = self.track.find_next_village_ahead(train_ahead.position_km)
                            
                            if side_track and side_track.occupied_side_tracks < side_track.side_tracks:
                                print(f"\n🛤  OVERTAKING STEP 2: Moving {train_ahead.name} to yellow track at {side_track.name}")
                                
                                train_ahead.move_to_side_track(side_track, f"Allowing {train_behind.name} to overtake")
                                train_ahead.waiting_for_train = train_behind
                                
                                # Step 3: Restore faster train's speed
                                print(f"\n⚡ OVERTAKING STEP 3: {train_behind.name} resuming full speed")
                                train_behind.resume_normal_speed()
                                
                                self.overtaking_events.append({
                                    'slower_train': train_ahead.name,
                                    'faster_train': train_behind.name,
                                    'location': side_track.name,
                                    'time': self.simulation_minutes
                                })
    
    def _process_side_track_returns(self):
        """Process trains returning from side tracks"""
        for train in self.trains:
            if train.is_on_side_track and train.waiting_for_train and train.side_track_location:
                faster_train = train.waiting_for_train
                side_track_pos = train.side_track_location.position_km
                faster_train_pos = faster_train.position_km
                buffer_distance = 30  # 30km safety buffer
                
                # Check if faster train has passed with buffer
                required_position = side_track_pos + buffer_distance
                has_passed = (faster_train_pos > required_position or faster_train.destination_reached)
                
                if has_passed:
                    # Check if main track is clear
                    main_track_clear = self._is_main_track_clear_at_position(side_track_pos, 15)
                    
                    if main_track_clear:
                        print(f"\n🔄 STEP 4: {train.name} returning from {train.side_track_location.name}")
                        train.return_to_main_track()
    
    def _is_main_track_clear_at_position(self, position_km, buffer_km=20):
        """Check if main track is clear at given position with buffer"""
        for train in self.trains:
            if (train.has_started and 
                not train.destination_reached and 
                not train.is_on_side_track and
                abs(train.position_km - position_km) < buffer_km):
                return False
        return True
    
    def _update_track_occupancy(self):
        """Update which trains are in which segments"""
        # Clear all segments
        for segment in self.track.segments:
            segment['trains'] = []
        
        # Add active trains to segments
        for train in self.trains:
            if train.has_started and not train.destination_reached and not train.is_on_side_track:
                segment = self.track.get_segment_at_position(train.position_km)
                if segment:
                    segment['trains'].append(train)
    
    def get_system_status(self):
        """Get comprehensive system status with chain reaction tracking"""
        active_trains = [t for t in self.trains if t.has_started and not t.destination_reached]
        
        # Count trains in each phase
        phase_counts = {}
        for train in active_trains:
            phase = train.current_phase
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        # Count primary vs secondary delays
        primary_delays = len([t for t in active_trains if t.delay_minutes > 0 and t.phase_target_train is None])
        secondary_delays = len([t for t in active_trains if t.delay_minutes > 0 and t.phase_target_train is not None])
        
        return {
            'current_time': self.current_time.strftime('%H:%M:%S'),
            'simulation_minutes': self.simulation_minutes,
            'total_trains': len(self.trains),
            'waiting_to_start': len([t for t in self.trains if not t.has_started]),
            'active_trains': len(active_trains),
            'primary_delays': primary_delays,
            'secondary_delays': secondary_delays,
            'total_delayed_trains': len([t for t in active_trains if t.delay_minutes > 0 or t.total_delay_accumulated > 0]),
            'side_track_trains': len([t for t in active_trains if t.is_on_side_track]),
            'emergency_stopped': len([t for t in active_trains if t.emergency_stopped]),
            'completed_trains': len([t for t in self.trains if t.destination_reached]),
            'overtaking_events': len(self.overtaking_events),
            'phase_counts': phase_counts
        }

class RailwayVisualizer:
    """Visual interface for the 4-phase railway system"""
    
    def __init__(self, width=1600, height=1000):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("4-Phase Railway Delay Handling with Emergency Stop")
        self.clock = pygame.time.Clock()
        
        # Initialize fonts
        self.font_small = pygame.font.Font(None, 18)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 32)
        
        # Layout constants
        self.track_y = 200
        self.track_height = 300
        self.side_track_spacing = 25
        
    def run_simulation(self):
        """Run the simulation with 4-phase delay logic"""
        scheduler = DynamicRailwayScheduler()
        
        # Define trains with variable speeds
        train_configs = [
            {'id': 1, 'name': 'Rajdhani Express', 'priority': 1, 'speed': 130},
            {'id': 2, 'name': 'Shatabdi Express', 'priority': 1, 'speed': 125},
            {'id': 3, 'name': 'Duronto Express', 'priority': 2, 'speed': 110},
            {'id': 4, 'name': 'Mail Express', 'priority': 2, 'speed': 100},
            {'id': 5, 'name': 'Passenger Train', 'priority': 3, 'speed': 80},
            {'id': 6, 'name': 'Local Train', 'priority': 3, 'speed': 70},
            {'id': 7, 'name': 'Goods Train Fast', 'priority': 4, 'speed': 65},
            {'id': 8, 'name': 'Goods Train Slow', 'priority': 4, 'speed': 50}
        ]
        
        scheduler.create_dynamic_schedule(train_configs)
        
        running = True
        paused = False
        simulation_speed = 0.05
        
        print("\n" + "="*80)
        print("4-PHASE RAILWAY DELAY HANDLING WITH EMERGENCY STOP")
        print("Press D to trigger delays, SPACE to pause, R to reset")
        print("Phase 4: Emergency Stop at ≤10km with speed = 0")
        print("="*80)
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_d:
                        delayed_train = scheduler.trigger_user_delay()
                        if delayed_train:
                            print(f"User triggered delay on {delayed_train.name}")
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print("Simulation paused" if paused else "Simulation resumed")
                    elif event.key == pygame.K_r:
                        scheduler = DynamicRailwayScheduler()
                        scheduler.create_dynamic_schedule(train_configs)
                        print("Simulation reset")
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            # Update simulation
            if not paused:
                scheduler.simulate_step(simulation_speed)
            
            # Clear screen
            self.screen.fill((245, 245, 220))
            
            # Draw everything
            self._draw_title_and_time(scheduler)
            self._draw_railway_network(scheduler)
            self._draw_system_status(scheduler)
            self._draw_train_details(scheduler)
            self._draw_phase_rules_panel()
            self._draw_controls()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)
            
        pygame.quit()
        sys.exit()
    
    def _draw_title_and_time(self, scheduler):
        """Draw title and current time"""
        title = self.font_large.render("4-Phase Railway Delay Handling with Emergency Stop", True, (0, 0, 100))
        self.screen.blit(title, (50, 20))
        
        status = scheduler.get_system_status()
        time_text = self.font_medium.render(f"Time: {status['current_time']} | Simulation: {status['simulation_minutes']:.1f} min", True, (100, 0, 0))
        self.screen.blit(time_text, (50, 50))
    
    def _draw_railway_network(self, scheduler):
        """Draw the railway network with trains"""
        track = scheduler.track
        
        # Draw track segments
        for segment in track.segments:
            start_x = self._position_to_pixel(segment['start_km'])
            end_x = self._position_to_pixel(segment['end_km'])
            
            if segment['has_double_track']:
                # Double track - two parallel lines
                pygame.draw.line(self.screen, (0, 0, 0), (start_x, self.track_y - 8), (end_x, self.track_y - 8), 4)
                pygame.draw.line(self.screen, (0, 0, 0), (start_x, self.track_y + 8), (end_x, self.track_y + 8), 4)
                # Label
                mid_x = (start_x + end_x) // 2
                label = self.font_small.render("Double", True, (0, 100, 0))
                self.screen.blit(label, (mid_x - 15, self.track_y - 35))
            else:
                # Single track
                pygame.draw.line(self.screen, (150, 0, 0), (start_x, self.track_y), (end_x, self.track_y), 5)
                # Label
                mid_x = (start_x + end_x) // 2
                label = self.font_small.render("Single", True, (150, 0, 0))
                self.screen.blit(label, (mid_x - 15, self.track_y + 25))
        
        # Draw locations and side tracks
        for location in track.locations:
            x = self._position_to_pixel(location.position_km)
            
            # Location marker
            size = 15 if location.type == 'city' else (12 if location.type == 'town' else 8)
            color = (0, 0, 200) if location.type == 'city' else ((0, 150, 0) if location.type == 'town' else (100, 100, 0))
            pygame.draw.circle(self.screen, color, (x, self.track_y), size)
            
            # Location name
            name = self.font_small.render(location.name, True, (0, 0, 0))
            self.screen.blit(name, (x - 25, self.track_y - 50))
            
            # Side tracks (yellow tracks)
            if location.side_tracks > 0:
                for i in range(location.side_tracks):
                    side_y = self.track_y + 40 + (i * self.side_track_spacing)
                    pygame.draw.line(self.screen, (255, 255, 0), (x - 25, side_y), (x + 25, side_y), 3)
                    # Connection lines
                    pygame.draw.line(self.screen, (150, 150, 0), (x - 15, self.track_y + 15), (x - 15, side_y), 2)
                    pygame.draw.line(self.screen, (150, 150, 0), (x + 15, self.track_y + 15), (x + 15, side_y), 2)
                
                # Side track status
                status_text = f"{location.occupied_side_tracks}/{location.side_tracks}"
                status_color = (255, 0, 0) if location.occupied_side_tracks == location.side_tracks else (0, 100, 0)
                status = self.font_small.render(status_text, True, status_color)
                self.screen.blit(status, (x - 10, self.track_y + 40 + location.side_tracks * self.side_track_spacing))
        
        # Draw trains
        for train in scheduler.trains:
            if train.has_started:
                self._draw_train(train)
    
    def _draw_train(self, train):
        """Draw individual train with 4-phase status indicators"""
        x = self._position_to_pixel(train.position_km)
        
        if train.is_on_side_track:
            # Draw on yellow side track - STATIONARY
            side_track_index = 0
            y = self.track_y + 40 + (side_track_index * self.side_track_spacing)
        else:
            y = self.track_y
        
        # Train size based on priority
        size = 18 if train.priority == 1 else (15 if train.priority == 2 else 12)
        
        # Train body
        pygame.draw.circle(self.screen, train.color, (x, y), size)
        pygame.draw.circle(self.screen, (255, 255, 255), (x, y), size, 3)
        
        # Priority number
        priority_text = self.font_small.render(str(train.priority), True, (255, 255, 255))
        self.screen.blit(priority_text, (x - 5, y - 6))
        
        # Phase-specific visual indicators
        if train.current_phase == "EMERGENCY":
            # Red flashing border for emergency stop
            if pygame.time.get_ticks() % 400 < 200:
                pygame.draw.circle(self.screen, (255, 0, 0), (x, y), size + 8, 5)
            # Additional emergency indicator
            pygame.draw.rect(self.screen, (255, 0, 0), (x - 4, y - 4, 8, 8))
            
        elif train.current_phase == "SPEED_MATCH":
            # Blue indicator for speed matching
            pygame.draw.circle(self.screen, (0, 150, 255), (x, y), size + 6, 3)
            
        elif train.current_phase == "PROGRESSIVE":
            # Orange indicator for progressive slowdown
            pygame.draw.circle(self.screen, (255, 165, 0), (x, y), size + 5, 3)
            
        elif train.current_phase == "MONITOR":
            # Green indicator for monitoring
            pygame.draw.circle(self.screen, (0, 200, 0), (x, y), size + 4, 2)
        
        if train.is_on_side_track:
            # Yellow indicator for side track
            pygame.draw.circle(self.screen, (255, 255, 0), (x, y), size + 3, 4)
            pygame.draw.rect(self.screen, (255, 255, 0), (x - 3, y - 3, 6, 6))
        
        # Train name
        name = self.font_small.render(train.name[:12], True, (0, 0, 0))
        self.screen.blit(name, (x - 35, y - 35))
        
        # Speed and status with phase information
        if train.destination_reached:
            status = "COMPLETE"
            status_color = (0, 150, 0)
        elif train.is_on_side_track:
            status = "YELLOW-STOP"
            status_color = (255, 140, 0)
        elif train.delay_minutes > 0:
            status = f"DELAY: {train.delay_minutes:.0f}m"
            status_color = (255, 0, 0)
        elif train.current_phase == "EMERGENCY":
            status = f"PHASE 4: STOP (0km/h)"
            status_color = (255, 0, 0)
        elif train.current_phase == "SPEED_MATCH":
            status = f"PHASE 3: {train.current_speed:.0f}km/h"
            status_color = (0, 100, 255)
        elif train.current_phase == "PROGRESSIVE":
            status = f"PHASE 2: {train.current_speed:.0f}km/h"
            status_color = (255, 165, 0)
        elif train.current_phase == "MONITOR":
            status = f"PHASE 1: {train.current_speed:.0f}km/h"
            status_color = (0, 150, 0)
        else:
            status = f"{train.current_speed}km/h"
            status_color = (0, 100, 0)
        
        status_text = self.font_small.render(status, True, status_color)
        self.screen.blit(status_text, (x - 30, y + 25))
    
    def _draw_system_status(self, scheduler):
        """Draw system status panel with chain reaction information"""
        status = scheduler.get_system_status()
        
        panel_x, panel_y = 50, 550
        panel_width, panel_height = 350, 200
        
        pygame.draw.rect(self.screen, (255, 255, 255), (panel_x, panel_y, panel_width, panel_height), border_radius=10)
        pygame.draw.rect(self.screen, (0, 0, 0), (panel_x, panel_y, panel_width, panel_height), 2, border_radius=10)
        
        title = self.font_medium.render("SYSTEM STATUS", True, (0, 100, 200))
        self.screen.blit(title, (panel_x + 15, panel_y + 15))
        
        status_items = [
            f"Total Trains: {status['total_trains']}",
            f"Waiting to Start: {status['waiting_to_start']}",
            f"Active Trains: {status['active_trains']}",
            f"Primary Delays: {status['primary_delays']}",
            f"Chain Reaction Delays: {status['secondary_delays']}",
            f"Emergency Stopped: {status['emergency_stopped']}",
            f"On Yellow Tracks: {status['side_track_trains']}",
            f"Completed: {status['completed_trains']}",
            f"Overtaking Events: {status['overtaking_events']}"
        ]
        
        for i, item in enumerate(status_items):
            color = (0, 0, 0)
            if "Primary Delays" in item and status['primary_delays'] > 0:
                color = (255, 0, 0)
            elif "Chain Reaction" in item and status['secondary_delays'] > 0:
                color = (255, 100, 0)  # Orange for chain reactions
            elif "Emergency" in item and status['emergency_stopped'] > 0:
                color = (255, 0, 0)
            elif "Yellow Tracks" in item and status['side_track_trains'] > 0:
                color = (255, 140, 0)
            elif "Completed" in item:
                color = (0, 150, 0)
            
            text = self.font_small.render(item, True, color)
            self.screen.blit(text, (panel_x + 20, panel_y + 45 + i * 20))
    
    def _draw_train_details(self, scheduler):
        """Draw detailed train information with phases"""
        panel_x, panel_y = 450, 550
        panel_width, panel_height = 400, 200
        
        pygame.draw.rect(self.screen, (255, 255, 255), (panel_x, panel_y, panel_width, panel_height), border_radius=10)
        pygame.draw.rect(self.screen, (0, 0, 0), (panel_x, panel_y, panel_width, panel_height), 2, border_radius=10)
        
        title = self.font_medium.render("TRAIN DETAILS", True, (0, 100, 200))
        self.screen.blit(title, (panel_x + 15, panel_y + 15))
        
        active_trains = [t for t in scheduler.trains if t.has_started and not t.destination_reached]
        
        if active_trains:
            for i, train in enumerate(active_trains[:6]):
                y_pos = panel_y + 45 + i * 25
                
                # Train name and priority with color
                name_text = f"{train.name[:12]} (P{train.priority})"
                name_surface = self.font_small.render(name_text, True, train.color)
                self.screen.blit(name_surface, (panel_x + 15, y_pos))
                
                # Position
                pos_text = f"{train.position_km:.1f}km"
                pos_surface = self.font_small.render(pos_text, True, (0, 0, 0))
                self.screen.blit(pos_surface, (panel_x + 150, y_pos))
                
                # Speed with phase info
                if train.current_phase == "EMERGENCY":
                    speed_text = f"0km/h (STOP)"
                    speed_color = (255, 0, 0)
                elif train.current_speed != train.original_speed:
                    speed_text = f"{train.current_speed:.0f}/{train.original_speed}km/h"
                    if train.current_phase == "PROGRESSIVE":
                        speed_color = (255, 165, 0)
                    elif train.current_phase == "SPEED_MATCH":
                        speed_color = (0, 100, 255)
                    else:
                        speed_color = (255, 140, 0)
                else:
                    speed_text = f"{train.current_speed:.0f}km/h"
                    speed_color = (0, 100, 0)
                
                speed_surface = self.font_small.render(speed_text, True, speed_color)
                self.screen.blit(speed_surface, (panel_x + 200, y_pos))
                
                # Phase status
                if train.is_on_side_track:
                    status = "YELLOW"
                    status_color = (255, 140, 0)
                elif train.current_phase == "EMERGENCY":
                    status = "PHASE 4"
                    status_color = (255, 0, 0)
                elif train.current_phase == "SPEED_MATCH":
                    status = "PHASE 3"
                    status_color = (0, 100, 255)
                elif train.current_phase == "PROGRESSIVE":
                    status = "PHASE 2"
                    status_color = (255, 165, 0)
                elif train.current_phase == "MONITOR":
                    status = "PHASE 1"
                    status_color = (0, 150, 0)
                elif train.delay_minutes > 0:
                    status = f"D:{train.delay_minutes:.0f}m"
                    status_color = (255, 0, 0)
                else:
                    status = "NORMAL"
                    status_color = (0, 100, 0)
                
                status_surface = self.font_small.render(status, True, status_color)
                self.screen.blit(status_surface, (panel_x + 310, y_pos))
        else:
            no_trains = self.font_small.render("No active trains", True, (100, 100, 100))
            self.screen.blit(no_trains, (panel_x + 20, panel_y + 50))
    
    def _draw_phase_rules_panel(self):
        """Draw the universal 4-phase rules with chain reaction logic"""
        panel_x, panel_y = 900, 550
        panel_width, panel_height = 350, 200
        
        pygame.draw.rect(self.screen, (255, 255, 255), (panel_x, panel_y, panel_width, panel_height), border_radius=10)
        pygame.draw.rect(self.screen, (0, 0, 0), (panel_x, panel_y, panel_width, panel_height), 2, border_radius=10)
        
        title = self.font_medium.render("UNIVERSAL + CHAIN REACTIONS", True, (0, 100, 200))
        self.screen.blit(title, (panel_x + 15, panel_y + 15))
        
        rules = [
            "UNIVERSAL 4-PHASE LOGIC:",
            "• Works for ALL speed relationships",
            "• Distance-based, not speed-based",
            "",
            "CHAIN REACTION SYSTEM:",
            "• Train A delays → Train B delays",
            "• Train B delays → Train C delays",  
            "• Cascading effect through network",
            "",
            "SPEED RECOVERY FIX:",
            "• Delayed trains resume original speed",
            "• Secondary delays recover properly",
            "• Full state reset on recovery",
            "",
            "PHASES (50→35→20→10→0km):",
            "Green→Orange→Blue→Red Flash",
            "Monitor→Progressive→Match→Stop"
        ]
        
        for i, rule in enumerate(rules):
            if rule.endswith(":"):
                color = (100, 0, 100)
            elif rule.startswith("•"):
                color = (0, 100, 0)
            elif "Green→Orange→Blue→Red" in rule:
                color = (150, 0, 150)
            elif rule == "":
                continue
            else:
                color = (0, 0, 0)
            
            text = self.font_small.render(rule, True, color)
            self.screen.blit(text, (panel_x + 15, panel_y + 45 + i * 11))
    
    def _draw_controls(self):
        """Draw control instructions"""
        panel_x, panel_y = 1300, 550
        panel_width, panel_height = 250, 200
        
        pygame.draw.rect(self.screen, (255, 255, 255), (panel_x, panel_y, panel_width, panel_height), border_radius=10)
        pygame.draw.rect(self.screen, (0, 0, 0), (panel_x, panel_y, panel_width, panel_height), 2, border_radius=10)
        
        title = self.font_medium.render("CONTROLS", True, (0, 100, 200))
        self.screen.blit(title, (panel_x + 15, panel_y + 15))
        
        controls = [
            "D - Trigger Random Delay",
            "   (Test universal logic)",
            "SPACE - Pause/Resume",
            "R - Reset Simulation",
            "ESC - Quit",
            "",
            "UNIVERSAL SYSTEM:",
            "• Fast trains slow for delays",
            "• Slow trains slow for delays", 
            "• Equal speed trains slow too",
            "• ANY train → Phase 4 at ≤10km",
            "• Speed relationship ignored",
            "",
            "PHASE INDICATORS:",
            "Green = Monitor (Phase 1)",
            "Orange = Progressive (Phase 2)", 
            "Blue = Speed Match (Phase 3)",
            "Red Flash = Emergency (Phase 4)"
        ]
        
        for i, control in enumerate(controls):
            if control.startswith('•'):
                color = (0, 100, 0)
            elif control.startswith('   '):
                color = (100, 100, 100)
            elif control in ["UNIVERSAL SYSTEM:", "PHASE INDICATORS:"]:
                color = (100, 0, 100)
            elif "=" in control and any(word in control for word in ["Green", "Orange", "Blue", "Red"]):
                if "Green" in control:
                    color = (0, 150, 0)
                elif "Orange" in control:
                    color = (255, 165, 0)
                elif "Blue" in control:
                    color = (0, 100, 255)
                elif "Red" in control:
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 0)
            elif control == "":
                continue
            else:
                color = (0, 0, 0)
            
            text = self.font_small.render(control, True, color)
            self.screen.blit(text, (panel_x + 15, panel_y + 45 + i * 10))
    
    def _position_to_pixel(self, position_km):
        """Convert km position to pixel coordinate"""
        track_width = self.width - 100
        return int(50 + (position_km / 400) * track_width)

# Main execution
if __name__ == "__main__":
    print("="*80)
    print("UNIVERSAL 4-PHASE RAILWAY DELAY HANDLING")
    print("Applies to ALL Speed Relationships - Fast, Slow, Equal")
    print("="*80)
    print()
    print("UNIVERSAL 4-PHASE LOGIC:")
    print("REMOVED SPEED CONDITION - Works for ANY approaching train!")
    print()
    print("Phase 1 (50-35km): Monitor Only")
    print("  → ALL trains maintain original speed when far")
    print("  → Fast trains, slow trains, equal speed trains")
    print("  → Visual: Green ring indicator")
    print()
    print("Phase 2 (35-20km): Progressive Slowdown") 
    print("  → ALL trains slow down progressively")
    print("  → Whether originally faster, slower, or equal")
    print("  → Visual: Orange ring indicator")
    print()
    print("Phase 3 (20-10km): Speed Matching")
    print("  → ALL trains match delayed train speed")
    print("  → Universal collision risk assessment")
    print("  → Visual: Blue ring indicator")
    print()
    print("Phase 4 (≤10km): Emergency Stop")
    print("  → ALL trains STOP (speed = 0)")
    print("  → Fast/slow/equal - doesn't matter!")
    print("  → Same delay inheritance for everyone")
    print("  → Visual: Red flashing indicator")
    print()
    print("KEY BREAKTHROUGH:")
    print("- No more 'if train_behind.speed > delayed_train.speed'")
    print("- Logic applies to trains approaching from ANY direction")
    print("- Distance-based phases work universally")
    print("- Complete collision prevention system")
    print()
    print("TEST SCENARIOS:")
    print("- Fast train → Slow delayed train")
    print("- Slow train → Fast delayed train") 
    print("- Equal speed trains")
    print("- Mixed priority combinations")
    print()
    print("CONTROLS:")
    print("- D: Trigger random delay (test all scenarios)")
    print("- SPACE: Pause/Resume simulation")
    print("- R: Reset simulation")
    print("- ESC: Quit")
    print()
    print("="*80)
    
    try:
        visualizer = RailwayVisualizer()
        visualizer.run_simulation()
    except KeyboardInterrupt:
        print("\nSimulation interrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            pygame.quit()
        except:
            pass
        print("Simulation ended.")