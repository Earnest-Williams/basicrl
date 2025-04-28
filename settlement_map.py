import random
import math
import os
from collections import deque

class Logger:
    def __init__(self, filename="test.txt"):
        self.filename = filename
        # Clear the file at startup
        with open(self.filename, 'w') as f:
            f.write("")
        
    def log(self, message):
        with open(self.filename, 'a') as f:
            f.write(f"{message}\n")
        
    def log_step(self, stage, message):
        with open(self.filename, 'a') as f:
            f.write(f"[{stage}] {message}\n")
    
    def log_error(self, message):
        print(f"ERROR: {message}")
        with open(self.filename, 'a') as f:
            f.write(f"ERROR: {message}\n")

# Initialize the logger
log = Logger()

class Rect:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
    def center(self):
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    def intersects(self, other):
        return (self.x < other.x + other.width and
                self.x + self.width > other.x and
                self.y < other.y + other.height and
                self.y + self.height > other.y)
                
    def contains(self, x, y):
        return (self.x <= x < self.x + self.width and
                self.y <= y < self.y + self.height)
    
    def distance_to(self, other):
        """Calculate the Manhattan distance between rectangles"""
        c1 = self.center()
        c2 = other.center()
        return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])
    
    def __repr__(self):
        return f"Rect(x={self.x}, y={self.y}, w={self.width}, h={self.height})"

class TileType:
    # Basic structure tiles
    EMPTY = ' '
    WALL = '#'
    DOOR = '+'
    CAVE_WALL = '%'
    
    # Housing block tiles
    HALLWAY = 'H'
    PARENT_BED = 'P'
    CHILD_BED = 'C'
    LIVING = 'L'
    KITCHEN = 'K'
    TOILET = 'T'
    WORKSHOP = 'W'
    STORAGE = 'S'
    
    # Community tiles
    SQUARE = '.'
    TEMPLE = 'M'  # M for Meeting hall/Temple
    TOWN_HALL = 'G'  # G for Government
    SHOP = '$'
    MUSHROOM_FARM = 'm'
    PATH = '.'

class HousingBlock:
    def __init__(self, block_id=0, size_factor=1.0):
        # Housing blocks now have variable sizes based on their importance in the tree
        base_size = 40
        size = int(base_size * size_factor)
        self.rect = Rect(0, 0, size, size)
        self.id = block_id
        self.width = self.rect.width
        self.height = self.rect.height
        self.map = [[TileType.EMPTY for _ in range(self.width)] for _ in range(self.height)]
        self.parent = None
        self.children = []
        self.depth = 0  # Depth in the tree (root = 0)
        log.log_step("HousingBlock", f"Created block {block_id} with size {size}x{size}")
    
    def generate_layout(self):
        log.log_step("HousingBlock", f"Generating layout for block {self.id} at {self.rect}")
        
        # Create outer walls
        for x in range(self.width):
            for y in range(self.height):
                if (x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1):
                    self.map[y][x] = TileType.WALL
        
        # Create hallway around the border
        for x in range(1, self.width-1):
            self.map[1][x] = TileType.HALLWAY
            self.map[self.height-2][x] = TileType.HALLWAY
        
        for y in range(1, self.height-1):
            self.map[y][1] = TileType.HALLWAY
            self.map[y][self.width-2] = TileType.HALLWAY
        
        # Place central toilet (medieval-style communal facility)
        toilet_width = max(6, min(self.width // 4, self.height // 4))
        toilet_x = self.width // 2 - toilet_width // 2
        toilet_y = self.height // 2 - toilet_width // 2
        
        log.log_step("HousingBlock", f"Placing central toilet at ({toilet_x}, {toilet_y}) with size {toilet_width}x{toilet_width}")
        
        for y in range(toilet_y, toilet_y + toilet_width):
            for x in range(toilet_x, toilet_x + toilet_width):
                if (x == toilet_x or x == toilet_x + toilet_width - 1 or 
                    y == toilet_y or y == toilet_y + toilet_width - 1):
                    self.map[y][x] = TileType.WALL
                else:
                    self.map[y][x] = TileType.TOILET
        
        # Add doors to toilet
        self.map[toilet_y][toilet_x + toilet_width//2] = TileType.DOOR
        self.map[toilet_y + toilet_width-1][toilet_x + toilet_width//2] = TileType.DOOR
        self.map[toilet_y + toilet_width//2][toilet_x] = TileType.DOOR
        self.map[toilet_y + toilet_width//2][toilet_x + toilet_width-1] = TileType.DOOR
        
        # Divide the remaining space into apartments
        self._create_apartments(toilet_x, toilet_y, toilet_width)
        
        log.log_step("HousingBlock", f"Finished layout for block {self.id}")
    
    def _create_apartments(self, toilet_x, toilet_y, toilet_width):
        log.log_step("HousingBlock", f"Creating apartments in block {self.id}")
        
        # Create apartments in the corners
        # Top-left apartment
        self._create_apartment(2, 2, toilet_x - 2, toilet_y - 2)
        
        # Top-right apartment
        self._create_apartment(toilet_x + toilet_width, 2, self.width - toilet_x - toilet_width - 2, toilet_y - 2)
        
        # Bottom-left apartment
        self._create_apartment(2, toilet_y + toilet_width, toilet_x - 2, self.height - toilet_y - toilet_width - 2)
        
        # Bottom-right apartment
        self._create_apartment(toilet_x + toilet_width, toilet_y + toilet_width, 
                             self.width - toilet_x - toilet_width - 2, 
                             self.height - toilet_y - toilet_width - 2)
    
    def _create_apartment(self, x, y, width, height):
        if width < 6 or height < 6:
            log.log_step("HousingBlock", f"Apartment at ({x}, {y}) is too small ({width}x{height}), skipping")
            return  # Too small for an apartment
        
        log.log_step("HousingBlock", f"Creating apartment at ({x}, {y}) with size {width}x{height}")
        
        # Create walls
        for ix in range(x, x + width):
            for iy in range(y, y + height):
                if (ix == x or ix == x + width - 1 or iy == y or iy == y + height - 1):
                    if self.map[iy][ix] != TileType.HALLWAY:  # Don't overwrite hallway
                        self.map[iy][ix] = TileType.WALL
        
        # Add a door to hallway
        if x == 2:  # Left side
            door_y = y + height // 2
            self.map[door_y][x] = TileType.DOOR
        elif y == 2:  # Top side
            door_x = x + width // 2
            self.map[y][door_x] = TileType.DOOR
        elif x + width == self.width - 2:  # Right side
            door_y = y + height // 2
            self.map[door_y][x + width - 1] = TileType.DOOR
        elif y + height == self.height - 2:  # Bottom side
            door_x = x + width // 2
            self.map[y + height - 1][door_x] = TileType.DOOR
        
        # Divide apartment into rooms
        self._divide_apartment(x+1, y+1, width-2, height-2)
    
    def _divide_apartment(self, x, y, width, height):
        # Determine how many rooms based on size
        area = width * height
        
        log.log_step("HousingBlock", f"Dividing apartment at ({x}, {y}) with area {area}")
        
        # For medieval-style housing, prioritize common areas and workshops
        if width > height:  # Split horizontally
            # Parent bedroom at one end
            bed_width = max(6, width // 4)  # Ensure adequate bedroom size
            self._create_room(x, y, bed_width, height, TileType.PARENT_BED)
            
            remaining_width = width - bed_width
            
            if area > 200:  # Very large apartment: add workshop and storage
                workshop_width = remaining_width // 3
                storage_width = remaining_width // 3
                living_width = remaining_width - workshop_width - storage_width
                
                self._create_room(x + bed_width, y, living_width, height, TileType.LIVING)
                self._create_room(x + bed_width + living_width, y, workshop_width, height, TileType.WORKSHOP)
                self._create_room(x + bed_width + living_width + workshop_width, y, storage_width, height, TileType.STORAGE)
            elif area > 100:  # Medium apartment: add workshop
                workshop_width = remaining_width // 2
                living_width = remaining_width - workshop_width
                
                self._create_room(x + bed_width, y, living_width, height, TileType.LIVING)
                self._create_room(x + bed_width + living_width, y, workshop_width, height, TileType.WORKSHOP)
            else:  # Small apartment: just living space
                self._create_room(x + bed_width, y, remaining_width, height, TileType.LIVING)
        else:  # Split vertically
            # Parent bedroom at top
            bed_height = max(6, height // 4)
            self._create_room(x, y, width, bed_height, TileType.PARENT_BED)
            
            remaining_height = height - bed_height
            
            if area > 200:  # Very large apartment
                workshop_height = remaining_height // 3
                storage_height = remaining_height // 3
                living_height = remaining_height - workshop_height - storage_height
                
                self._create_room(x, y + bed_height, width, living_height, TileType.LIVING)
                self._create_room(x, y + bed_height + living_height, width, workshop_height, TileType.WORKSHOP)
                self._create_room(x, y + bed_height + living_height + workshop_height, width, storage_height, TileType.STORAGE)
            elif area > 100:  # Medium apartment
                kitchen_height = remaining_height // 2
                living_height = remaining_height - kitchen_height
                
                self._create_room(x, y + bed_height, width, living_height, TileType.LIVING)
                self._create_room(x, y + bed_height + living_height, width, kitchen_height, TileType.KITCHEN)
            else:  # Small apartment
                self._create_room(x, y + bed_height, width, remaining_height, TileType.LIVING)
    
    def _create_room(self, x, y, width, height, room_type):
        log.log_step("HousingBlock", f"Creating {room_type} room at ({x}, {y}) with size {width}x{height}")
        
        # Add internal walls for the room
        for iy in range(y, y + height):
            for ix in range(x, x + width):
                if 0 <= ix < self.width and 0 <= iy < self.height:
                    if (ix == x or ix == x + width - 1 or iy == y or iy == y + height - 1):
                        # Don't overwrite existing doors or hallways
                        if self.map[iy][ix] != TileType.DOOR and self.map[iy][ix] != TileType.HALLWAY:
                            self.map[iy][ix] = TileType.WALL
                    else:
                        self.map[iy][ix] = room_type
        
        # Add a door to adjacent rooms or hallway
        # For simplicity, we'll add doors on a fixed position
        if width > height:  # Wider room - door on vertical walls
            door_y = y + height // 2
            
            # Check if left side is adjacent to another room or hallway
            if x > 1 and (self.map[door_y][x-1] == TileType.HALLWAY or 
                         self.map[door_y][x-1] == TileType.LIVING or 
                         self.map[door_y][x-1] == TileType.KITCHEN):
                self.map[door_y][x] = TileType.DOOR
            
            # Check if right side is adjacent to another room or hallway
            elif x + width < self.width - 1 and (self.map[door_y][x+width] == TileType.HALLWAY or 
                                               self.map[door_y][x+width] == TileType.LIVING or 
                                               self.map[door_y][x+width] == TileType.KITCHEN):
                self.map[door_y][x+width-1] = TileType.DOOR
        else:  # Taller room - door on horizontal walls
            door_x = x + width // 2
            
            # Check if top is adjacent to another room or hallway
            if y > 1 and (self.map[y-1][door_x] == TileType.HALLWAY or 
                         self.map[y-1][door_x] == TileType.LIVING or 
                         self.map[y-1][door_x] == TileType.KITCHEN):
                self.map[y][door_x] = TileType.DOOR
            
            # Check if bottom is adjacent to another room or hallway
            elif y + height < self.height - 1 and (self.map[y+height][door_x] == TileType.HALLWAY or 
                                                 self.map[y+height][door_x] == TileType.LIVING or 
                                                 self.map[y+height][door_x] == TileType.KITCHEN):
                self.map[y+height-1][door_x] = TileType.DOOR

class CommunityBuilding:
    def __init__(self, rect, building_type):
        self.rect = rect
        self.type = building_type
        self.width = rect.width
        self.height = rect.height
        self.map = [[TileType.EMPTY for _ in range(rect.width)] for _ in range(rect.height)]
        log.log_step("CommunityBuilding", f"Created {building_type} building at {rect}")
    
    def generate_layout(self):
        log.log_step("CommunityBuilding", f"Generating layout for {self.type} building")
        
        # Create walls
        for x in range(self.width):
            for y in range(self.height):
                if (x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1):
                    self.map[y][x] = TileType.WALL
                else:
                    self.map[y][x] = self.type
        
        # Add door
        if self.type == TileType.TEMPLE:
            # Door at the bottom for temple
            door_x = self.width // 2
            door_y = self.height - 1
        elif self.type == TileType.TOWN_HALL:
            # Door at the top for town hall
            door_x = self.width // 2
            door_y = 0
        else:  # Shop or Mushroom Farm
            # Door on a random side
            side = random.randint(0, 3)
            if side == 0:  # Top
                door_x = self.width // 2
                door_y = 0
            elif side == 1:  # Right
                door_x = self.width - 1
                door_y = self.height // 2
            elif side == 2:  # Bottom
                door_x = self.width // 2
                door_y = self.height - 1
            else:  # Left
                door_x = 0
                door_y = self.height // 2
            
        self.map[door_y][door_x] = TileType.DOOR
        log.log_step("CommunityBuilding", f"Added door at ({door_x}, {door_y})")

class TreeBasedSettlement:
    def __init__(self):
        self.housing_blocks = []
        self.town_square = None
        self.temple = None
        self.town_hall = None
        self.shops = []
        
        # Map will be sized dynamically based on structure placement
        self.map = None
        self.width = 0
        self.height = 0
        
        log.log_step("Settlement", "Initialized tree-based settlement generator")
    
    def generate(self):
        log.log_step("Settlement", "Starting generation process")
        
        # 1. Generate tree of housing blocks
        self._generate_housing_block_tree()
        
        # 2. Assign depths and calculate sizes based on tree hierarchy
        self._assign_tree_depths()
        
        # 3. Generate layouts for all housing blocks
        self._generate_block_layouts()
        
        # 4. Position housing blocks based on tree structure
        self._position_housing_blocks_by_tree()
        
        # 5. Generate community buildings
        self._generate_community_buildings()
        
        # 6. Determine map bounds
        self._determine_map_bounds()
        
        # 7. Initialize map with cave walls
        self._initialize_map()
        
        # 8. Apply structures to map
        self._apply_structures()
        
        # 9. Create paths connecting everything
        self._create_paths()
        
        log.log_step("Settlement", "Generation complete")
        
        # Return settlement metrics
        return {
            "housing_blocks": len(self.housing_blocks),
            "map_size": (self.width, self.height)
        }
    
    def _generate_housing_block_tree(self):
        """Generate a tree of housing blocks using the algorithm specified"""
        log.log_step("Settlement", "Generating housing block tree")
        
        # Implementation of the algorithm from the specified pseudocode
        blocks = 1  # Start with 1 block always
        stage = 1   # Try number within a burst
        restart_bonus = 0  # +1% added to all future base chances after each successful restart
        
        # Create root block (first block)
        root_block = HousingBlock(block_id=0, size_factor=1.2)  # Root is slightly larger
        self.housing_blocks.append(root_block)
        
        block_id = 1  # ID for the next block
        
        while True:
            # Calculate success chance for this stage, including scaling bonus
            base_y = 113.25 * math.exp(-0.27 * stage) + 0.57
            success_chance = base_y + restart_bonus
            
            log.log_step("Settlement", f"Stage {stage}, success chance: {success_chance}%")
            
            # Attempt to add a normal block
            if random.random() * 100 > success_chance:
                log.log_step("Settlement", f"Failed chance check at stage {stage}, ending block generation")
                break  # Failed to build this block, end generation
            
            # Successfully passed chance check, create a new block
            # Size factor decreases with ID to make later blocks smaller
            size_factor = max(0.7, 1.0 - (block_id * 0.05))
            new_block = HousingBlock(block_id=block_id, size_factor=size_factor)
            block_id += 1
            
            # Find parent: randomly choose from existing blocks
            parent = random.choice(self.housing_blocks)
            parent.children.append(new_block)
            new_block.parent = parent
            
            log.log_step("Settlement", f"Added block {new_block.id} with parent {parent.id}")
            self.housing_blocks.append(new_block)
            
            # Check for bonus blocks with decaying 25% system
            bonus_chance = 25.0
            while bonus_chance >= 0.5:
                log.log_step("Settlement", f"Bonus chance: {bonus_chance}%")
                if random.random() * 100 < bonus_chance:
                    # Add bonus block
                    bonus_size_factor = max(0.7, 1.0 - (block_id * 0.05))
                    bonus_block = HousingBlock(block_id=block_id, size_factor=bonus_size_factor)
                    block_id += 1
                    
                    # Connect to the most recently added block
                    latest_block = self.housing_blocks[-1]
                    latest_block.children.append(bonus_block)
                    bonus_block.parent = latest_block
                    
                    log.log_step("Settlement", f"Added bonus block {bonus_block.id} with parent {latest_block.id}")
                    self.housing_blocks.append(bonus_block)
                    bonus_chance *= 0.75  # Decay bonus chance for next bonus
                else:
                    log.log_step("Settlement", "Failed bonus chance, stopping bonus chain")
                    break  # Stop bonus chain
            
            # Now check recursion after stages 4, 5, or 6
            if stage in (4, 5, 6):
                if random.random() < 0.5:  # coin flip
                    restart_bonus += 1  # Every successful restart makes future growth easier
                    log.log_step("Settlement", f"Restarting at stage {stage}, new restart bonus: {restart_bonus}")
                    stage = 1  # Restart stage counter
                elif stage == 6:
                    log.log_step("Settlement", "Reached stage 6 and failed coin flip, ending block generation")
                    break  # Auto-end if at stage 6 and tails
                else:
                    log.log_step("Settlement", f"Failed coin flip at stage {stage}, continuing to next stage")
                    stage += 1  # Push to next stage if tails and stage 4 or 5
            else:
                log.log_step("Settlement", f"Advancing from stage {stage} to {stage+1}")
                stage += 1  # Simple increment until hitting 4
        
        log.log_step("Settlement", f"Generated {len(self.housing_blocks)} housing blocks in tree")
    
    def _assign_tree_depths(self):
        """Assign depth values to all housing blocks based on their position in the tree"""
        log.log_step("Settlement", "Assigning tree depths to housing blocks")
        
        # Start with the root block (depth 0)
        root_block = self.housing_blocks[0]
        root_block.depth = 0
        
        # Use breadth-first search to assign depths
        queue = deque([root_block])
        while queue:
            block = queue.popleft()
            for child in block.children:
                child.depth = block.depth + 1
                queue.append(child)
        
        # Log the depth distribution
        max_depth = max(block.depth for block in self.housing_blocks)
        log.log_step("Settlement", f"Tree depth: {max_depth}")
        for depth in range(max_depth + 1):
            count = sum(1 for block in self.housing_blocks if block.depth == depth)
            log.log_step("Settlement", f"Depth {depth}: {count} blocks")
    
    def _generate_block_layouts(self):
        """Generate internal layouts for all housing blocks"""
        log.log_step("Settlement", "Generating internal layouts for all housing blocks")
        
        for block in self.housing_blocks:
            block.generate_layout()
    
    def _position_housing_blocks_by_tree(self):
        """Position housing blocks based on their tree structure"""
        log.log_step("Settlement", "Positioning housing blocks based on tree structure")
        
        # Start with the root at the origin
        root = self.housing_blocks[0]
        root.rect.x = 0
        root.rect.y = 0
        
        # Function to position a block's children around it
        def position_children(parent):
            # Calculate positions for children
            children = parent.children
            if not children:
                return
            
            # Calculate spacing based on parent size and child count
            angle_step = 2 * math.pi / len(children)
            radius = max(parent.rect.width, parent.rect.height) * 1.5
            
            # Place children in a circle around the parent
            for i, child in enumerate(children):
                # Calculate position using polar coordinates
                angle = i * angle_step
                # Add some randomness to the angle and radius
                angle += random.uniform(-0.2, 0.2)
                actual_radius = radius * random.uniform(0.8, 1.2)
                
                x = parent.rect.center()[0] + int(actual_radius * math.cos(angle))
                y = parent.rect.center()[1] + int(actual_radius * math.sin(angle))
                
                # Adjust to position the center of the child
                x -= child.rect.width // 2
                y -= child.rect.height // 2
                
                child.rect.x = x
                child.rect.y = y
                
                log.log_step("Settlement", f"Positioned block {child.id} at ({x}, {y})")
                
                # Recursively position this child's children
                position_children(child)
        
        # Start positioning from the root
        position_children(root)
        
        # Resolve any overlaps by nudging blocks
        self._resolve_overlaps()
    
    def _resolve_overlaps(self):
        """Resolve any overlapping blocks by nudging them away from each other"""
        log.log_step("Settlement", "Resolving overlaps between housing blocks")
        
        max_iterations = 100
        iteration = 0
        overlap_found = True
        
        while overlap_found and iteration < max_iterations:
            overlap_found = False
            iteration += 1
            
            # Check each pair of blocks for overlap
            for i, block1 in enumerate(self.housing_blocks):
                for block2 in self.housing_blocks[i+1:]:
                    if block1.rect.intersects(block2.rect):
                        overlap_found = True
                        
                        # Calculate vector between centers
                        c1 = block1.rect.center()
                        c2 = block2.rect.center()
                        dx = c2[0] - c1[0]
                        dy = c2[1] - c1[1]
                        
                        # Normalize and scale
                        length = max(1, math.sqrt(dx*dx + dy*dy))
                        dx = int(dx / length * 10)
                        dy = int(dy / length * 10)
                        
                        # Move both blocks in opposite directions
                        block1.rect.x -= dx
                        block1.rect.y -= dy
                        block2.rect.x += dx
                        block2.rect.y += dy
                        
                        log.log_step("Settlement", f"Resolved overlap between blocks {block1.id} and {block2.id}")
        
        # Handle remaining overlaps with more aggressive separation
        if overlap_found:
            log.log_step("Settlement", "Using aggressive overlap resolution")
            
            # Create a map of block positions
            blocks_by_pos = {}
            for block in self.housing_blocks:
                blocks_by_pos[(block.rect.center()[0] // 50, block.rect.center()[1] // 50)] = block
            
            # Find non-overlapping positions for problematic blocks
            for block in self.housing_blocks:
                overlaps = any(b != block and block.rect.intersects(b.rect) for b in self.housing_blocks)
                if overlaps:
                    # Find a free position
                    for distance in range(50, 501, 50):
                        placed = False
                        for angle in range(0, 360, 30):
                            angle_rad = math.radians(angle)
                            center_x = int(block.rect.center()[0] + distance * math.cos(angle_rad))
                            center_y = int(block.rect.center()[1] + distance * math.sin(angle_rad))
                            
                            grid_pos = (center_x // 50, center_y // 50)
                            if grid_pos not in blocks_by_pos:
                                # Position is free, move the block there
                                block.rect.x = center_x - block.rect.width // 2
                                block.rect.y = center_y - block.rect.height // 2
                                blocks_by_pos[grid_pos] = block
                                placed = True
                                log.log_step("Settlement", f"Moved block {block.id} to ({block.rect.x}, {block.rect.y})")
                                break
                        
                        if placed:
                            break
    
    def _generate_community_buildings(self):
        """Generate community buildings based on the housing block arrangement"""
        log.log_step("Settlement", "Generating community buildings")
        
        # Find the center of mass of all housing blocks
        total_x = sum(b.rect.center()[0] for b in self.housing_blocks)
        total_y = sum(b.rect.center()[1] for b in self.housing_blocks)
        center_x = total_x // len(self.housing_blocks)
        center_y = total_y // len(self.housing_blocks)
        
        log.log_step("Settlement", f"Community center at ({center_x}, {center_y})")
        
        # Town square size depends on housing block count
        housing_count = len(self.housing_blocks)
        square_size = max(30, min(50, 25 + housing_count * 2))
        
        self.town_square = Rect(center_x - square_size // 2, center_y - square_size // 2, 
                               square_size, square_size)
        
        log.log_step("Settlement", f"Town square at {self.town_square}")
        
        # Temple north of the square
        temple_width = min(40, max(20, 15 + housing_count * 2))
        temple_height = min(30, max(15, 10 + housing_count * 1))
        
        temple_x = center_x - temple_width // 2
        temple_y = center_y - square_size // 2 - temple_height - 5
        
        temple_rect = Rect(temple_x, temple_y, temple_width, temple_height)
        self.temple = CommunityBuilding(temple_rect, TileType.TEMPLE)
        self.temple.generate_layout()
        
        # Town hall south of the square
        town_hall_width = min(40, max(20, 15 + housing_count * 2))
        town_hall_height = min(30, max(15, 10 + housing_count * 1))
        
        town_hall_x = center_x - town_hall_width // 2
        town_hall_y = center_y + square_size // 2 + 5
        
        town_hall_rect = Rect(town_hall_x, town_hall_y, town_hall_width, town_hall_height)
        self.town_hall = CommunityBuilding(town_hall_rect, TileType.TOWN_HALL)
        self.town_hall.generate_layout()
        
        # Shops around the square
        shop_count = max(2, min(housing_count, 6))
        shop_width = 12
        shop_height = 10
        
        shop_types = [TileType.SHOP, TileType.MUSHROOM_FARM]
        
        # Calculate shop positions in a circle around the town square
        for i in range(shop_count):
            angle = 2 * math.pi * i / shop_count
            # Offset angle to avoid exactly overlapping with temple and town hall
            angle += math.pi / shop_count
            
            # Position around the square with some distance
            distance = square_size / 2 + 15
            shop_center_x = center_x + int(distance * math.cos(angle))
            shop_center_y = center_y + int(distance * math.sin(angle))
            
            shop_x = shop_center_x - shop_width // 2
            shop_y = shop_center_y - shop_height // 2
            
            shop_rect = Rect(shop_x, shop_y, shop_width, shop_height)
            shop = CommunityBuilding(shop_rect, shop_types[i % len(shop_types)])
            shop.generate_layout()
            self.shops.append(shop)
            
            log.log_step("Settlement", f"Created shop at {shop_rect}")
    
    def _determine_map_bounds(self):
        """Determine the bounds of the map based on all structures"""
        log.log_step("Settlement", "Determining map bounds")
        
        # Find the bounding box of all structures
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')
        
        # Check housing blocks
        for block in self.housing_blocks:
            min_x = min(min_x, block.rect.x)
            max_x = max(max_x, block.rect.x + block.rect.width)
            min_y = min(min_y, block.rect.y)
            max_y = max(max_y, block.rect.y + block.rect.height)
        
        # Check town square
        min_x = min(min_x, self.town_square.x)
        max_x = max(max_x, self.town_square.x + self.town_square.width)
        min_y = min(min_y, self.town_square.y)
        max_y = max(max_y, self.town_square.y + self.town_square.height)
        
        # Check temple
        if self.temple:
            min_x = min(min_x, self.temple.rect.x)
            max_x = max(max_x, self.temple.rect.x + self.temple.rect.width)
            min_y = min(min_y, self.temple.rect.y)
            max_y = max(max_y, self.temple.rect.y + self.temple.rect.height)
        
        # Check town hall
        if self.town_hall:
            min_x = min(min_x, self.town_hall.rect.x)
            max_x = max(max_x, self.town_hall.rect.x + self.town_hall.rect.width)
            min_y = min(min_y, self.town_hall.rect.y)
            max_y = max(max_y, self.town_hall.rect.y + self.town_hall.rect.height)
        
        # Check shops
        for shop in self.shops:
            min_x = min(min_x, shop.rect.x)
            max_x = max(max_x, shop.rect.x + shop.rect.width)
            min_y = min(min_y, shop.rect.y)
            max_y = max(max_y, shop.rect.y + shop.rect.height)
        
        # Add padding
        padding = 10
        min_x = min_x - padding
        min_y = min_y - padding
        max_x += padding
        max_y += padding
        
        log.log_step("Settlement", f"Raw map bounds: ({min_x}, {min_y}) to ({max_x}, {max_y})")
        
        # Normalize to origin
        for block in self.housing_blocks:
            block.rect.x -= min_x
            block.rect.y -= min_y
        
        self.town_square.x -= min_x
        self.town_square.y -= min_y
        
        if self.temple:
            self.temple.rect.x -= min_x
            self.temple.rect.y -= min_y
        
        if self.town_hall:
            self.town_hall.rect.x -= min_x
            self.town_hall.rect.y -= min_y
        
        for shop in self.shops:
            shop.rect.x -= min_x
            shop.rect.y -= min_y
        
        # Set map dimensions
        self.width = int(max_x - min_x)
        self.height = int(max_y - min_y)
        
        log.log_step("Settlement", f"Final map dimensions: {self.width}x{self.height}")
    
    def _initialize_map(self):
        """Initialize the map with cave walls"""
        log.log_step("Settlement", "Initializing map with cave walls")
        
        # Create map filled with cave walls
        self.map = [[TileType.CAVE_WALL for _ in range(self.width)] for _ in range(self.height)]
    
    def _apply_structures(self):
        """Apply all structures to the map"""
        log.log_step("Settlement", "Applying structures to map")
        
        # Apply town square
        log.log_step("Settlement", f"Applying town square at {self.town_square}")
        for y in range(self.town_square.height):
            for x in range(self.town_square.width):
                map_x = self.town_square.x + x
                map_y = self.town_square.y + y
                
                if 0 <= map_x < self.width and 0 <= map_y < self.height:
                    self.map[map_y][map_x] = TileType.SQUARE
        
        # Apply housing blocks
        for block in self.housing_blocks:
            log.log_step("Settlement", f"Applying housing block {block.id} at {block.rect}")
            for y in range(block.rect.height):
                for x in range(block.rect.width):
                    map_x = block.rect.x + x
                    map_y = block.rect.y + y
                    
                    if (0 <= map_x < self.width and 0 <= map_y < self.height and 
                        0 <= y < len(block.map) and 0 <= x < len(block.map[0]) and
                        block.map[y][x] != TileType.EMPTY):
                        self.map[map_y][map_x] = block.map[y][x]
        
        # Apply temple
        if self.temple:
            log.log_step("Settlement", f"Applying temple at {self.temple.rect}")
            for y in range(self.temple.rect.height):
                for x in range(self.temple.rect.width):
                    map_x = self.temple.rect.x + x
                    map_y = self.temple.rect.y + y
                    
                    if (0 <= map_x < self.width and 0 <= map_y < self.height and 
                        0 <= y < len(self.temple.map) and 0 <= x < len(self.temple.map[0])):
                        self.map[map_y][map_x] = self.temple.map[y][x]
        
        # Apply town hall
        if self.town_hall:
            log.log_step("Settlement", f"Applying town hall at {self.town_hall.rect}")
            for y in range(self.town_hall.rect.height):
                for x in range(self.town_hall.rect.width):
                    map_x = self.town_hall.rect.x + x
                    map_y = self.town_hall.rect.y + y
                    
                    if (0 <= map_x < self.width and 0 <= map_y < self.height and 
                        0 <= y < len(self.town_hall.map) and 0 <= x < len(self.town_hall.map[0])):
                        self.map[map_y][map_x] = self.town_hall.map[y][x]
        
        # Apply shops
        for i, shop in enumerate(self.shops):
            log.log_step("Settlement", f"Applying shop {i} at {shop.rect}")
            for y in range(shop.rect.height):
                for x in range(shop.rect.width):
                    map_x = shop.rect.x + x
                    map_y = shop.rect.y + y
                    
                    if (0 <= map_x < self.width and 0 <= map_y < self.height and 
                        0 <= y < len(shop.map) and 0 <= x < len(shop.map[0])):
                        self.map[map_y][map_x] = shop.map[y][x]
    
    def _create_paths(self):
        """Create paths connecting all structures"""
        log.log_step("Settlement", "Creating paths between structures")
        
        # Get all structures that need paths
        structures = [(block.rect, block.id) for block in self.housing_blocks]
        structures.append((self.town_square, "town_square"))
        if self.temple:
            structures.append((self.temple.rect, "temple"))
        if self.town_hall:
            structures.append((self.town_hall.rect, "town_hall"))
        for i, shop in enumerate(self.shops):
            structures.append((shop.rect, f"shop_{i}"))
        
        # Build a minimum spanning tree to connect all structures
        # This ensures all structures are connected with minimal paths
        
        # Start with a list of all edges (connections between structures)
        edges = []
        for i in range(len(structures)):
            for j in range(i+1, len(structures)):
                rect1, id1 = structures[i]
                rect2, id2 = structures[j]
                distance = rect1.distance_to(rect2)
                edges.append((distance, i, j))
        
        # Sort edges by distance
        edges.sort()
        
        # Kruskal's algorithm to find minimum spanning tree
        parent = list(range(len(structures)))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            parent[find(x)] = find(y)
        
        # Connect structures using the minimum spanning tree
        for dist, i, j in edges:
            if find(i) != find(j):
                union(i, j)
                
                rect1, id1 = structures[i]
                rect2, id2 = structures[j]
                
                log.log_step("Settlement", f"Creating path from {id1} to {id2}")
                self._create_path(rect1, rect2)
    
    def _create_path(self, source, dest):
        """Create a path between two structures"""
        source_center = (source.x + source.width // 2, source.y + source.height // 2)
        dest_center = (dest.x + dest.width // 2, dest.y + dest.height // 2)
        
        log.log_step("Settlement", f"Creating path from {source_center} to {dest_center}")
        
        # Create a curve-like path by selecting intermediate points
        points = [source_center]
        
        # Add 0-2 intermediate points for a more natural path
        num_points = random.randint(0, 2)
        for _ in range(num_points):
            # Calculate a point somewhere between source and dest
            t = random.uniform(0.3, 0.7)  # Parameter along the path
            x = int(source_center[0] + t * (dest_center[0] - source_center[0]))
            y = int(source_center[1] + t * (dest_center[1] - source_center[1]))
            
            # Add some randomness to the point
            x += random.randint(-10, 10)
            y += random.randint(-10, 10)
            
            points.append((x, y))
        
        points.append(dest_center)
        
        # Draw paths between consecutive points
        for i in range(len(points) - 1):
            self._draw_path_segment(points[i], points[i + 1])
    
    def _draw_path_segment(self, start, end):
        """Draw a straight path segment between two points"""
        # Use Bresenham's line algorithm to draw a path
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            # Only draw if the point is within bounds and is a cave wall
            if 0 <= x0 < self.width and 0 <= y0 < self.height:
                if self.map[y0][x0] == TileType.CAVE_WALL:
                    self.map[y0][x0] = TileType.PATH
            
            if x0 == x1 and y0 == y1:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
    
    def print_map(self):
        """Print the map to the log file"""
        map_text = ""
        for row in self.map:
            map_text += ''.join(row) + "\n"
        log.log(map_text)

def get_legend():
    """Return a string with the legend for map symbols"""
    legend = "\nLegend:\n"
    legend += "# - Wall\n"
    legend += "% - Cave Wall\n"
    legend += "+ - Door\n"
    legend += ". - Path/Square\n"
    legend += "H - Hallway\n"
    legend += "P - Parent Bedroom\n"
    legend += "C - Child Bedroom\n"
    legend += "L - Living Area\n"
    legend += "K - Kitchen\n"
    legend += "T - Toilet\n"
    legend += "W - Workshop\n"
    legend += "S - Storage\n"
    legend += "M - Temple/Meeting Hall\n"
    legend += "G - Town Hall\n"
    legend += "$ - Shop\n"
    legend += "m - Mushroom Farm\n"
    legend += "  - Empty space\n"
    return legend

def main():
    log.log_step("Main", "Generating tree-based cave settlement...")
    
    # Set a seed for reproducibility
    random.seed(42)
    
    # Create and generate settlement
    settlement = TreeBasedSettlement()
    metrics = settlement.generate()
    
    # Print the map and legend
    settlement.print_map()
    log.log(get_legend())
    
    # Print metrics
    log.log(f"\nSettlement Metrics:")
    log.log(f"Housing Blocks: {metrics['housing_blocks']}")
    log.log(f"Map Size: {metrics['map_size'][0]}x{metrics['map_size'][1]}")

if __name__ == "__main__":
    main()
