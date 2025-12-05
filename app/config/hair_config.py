"""
Hair Style Prompts - SIMPLIFIED VERSION
10 KIỂU TÓC THỊNH HÀNH - Prompt ngắn gọn cho SD1.5
"""

from typing import Dict


class HairStylePrompts:
    """
    10 kiểu tóc thịnh hành - Prompt đơn giản, hiệu quả cho SD1.5
    """

    # ==================== BASE NEGATIVE ĐƠN GIẢN ====================
    BASE_NEGATIVE = (
        "different face, changed face, face swap, "
        "blurry face, distorted face, deformed face, "
        "ugly, bad anatomy, extra limbs, "
        "low quality, watermark, signature"
    )

    # ==================== 10 KIỂU TÓC THỊNH HÀNH ====================

    # 1. TÓC NGẮN MODERN (Undercut Fade)
    SHORT_UNDERCUT = {
        "id": "short_undercut",
        "name": "Undercut Fade",
        "prompt": "man with modern undercut fade hairstyle, shaved sides, sharp fade, trendy male haircut",
        "negative": "long hair, messy hair, " + BASE_NEGATIVE,
        "category": "short",
        "gender": "male"
    }

    # 2. TÓC BUZZ CUT
    BUZZ_CUT = {
        "id": "buzz_cut",
        "name": "Buzz Cut",
        "prompt": "man with buzz cut hairstyle, very short hair, military cut, clean shaved head",
        "negative": "long hair, medium hair, " + BASE_NEGATIVE,
        "category": "short",
        "gender": "male"
    }

    # 3. TÓC MAN BUN
    MAN_BUN = {
        "id": "man_bun",
        "name": "Man Bun",
        "prompt": "man with man bun hairstyle, long hair tied up in bun, top knot, masculine style",
        "negative": "short hair, loose hair, " + BASE_NEGATIVE,
        "category": "long",
        "gender": "male"
    }

    # 4. TÓC SIDEPART
    SIDE_PART = {
        "id": "side_part",
        "name": "Side Part",
        "prompt": "man with side part hairstyle, neat combed hair, professional haircut, classic style",
        "negative": "messy hair, no part, " + BASE_NEGATIVE,
        "category": "medium",
        "gender": "male"
    }

    # 5. TÓC SLICKED BACK
    SLICKED_BACK = {
        "id": "slicked_back",
        "name": "Slicked Back",
        "prompt": "man with slicked back hairstyle, smooth combed back hair, glossy finish, elegant style",
        "negative": "messy hair, forward hair, " + BASE_NEGATIVE,
        "category": "styled",
        "gender": "male"
    }

    # 6. TÓC CURLY AFRO
    CURLY_AFRO = {
        "id": "curly_afro",
        "name": "Curly Afro",
        "prompt": "person with curly afro hairstyle, natural curls, voluminous hair, textured hair",
        "negative": "straight hair, flat hair, " + BASE_NEGATIVE,
        "category": "curly",
        "gender": "unisex"
    }

    # 7. TÓC BOB CUT (Nữ)
    BOB_CUT = {
        "id": "bob_cut",
        "name": "Bob Cut",
        "prompt": "woman with bob haircut, shoulder length hair, sleek style, feminine cut",
        "negative": "long hair, short hair, " + BASE_NEGATIVE,
        "category": "medium",
        "gender": "female"
    }

    # 8. TÓC PIXIE CUT (Nữ)
    PIXIE_CUT = {
        "id": "pixie_cut",
        "name": "Pixie Cut",
        "prompt": "woman with pixie cut hairstyle, short layered crop, feminine short hair, chic style",
        "negative": "long hair, medium hair, " + BASE_NEGATIVE,
        "category": "short",
        "gender": "female"
    }

    # 9. TÓC MÀU XANH (Colored)
    BLUE_HAIR = {
        "id": "blue_hair",
        "name": "Blue Hair",
        "prompt": "person with blue hair color, vibrant blue hairstyle, fantasy hair color, colorful hair",
        "negative": "natural hair color, brown hair, black hair, " + BASE_NEGATIVE,
        "category": "colored",
        "gender": "unisex"
    }

    # 10. TÓC KOREAN STYLE
    KOREAN_STYLE = {
        "id": "korean_style",
        "name": "Korean Style",
        "prompt": "man with Korean hairstyle, K-pop style, textured fringe, Asian male haircut",
        "negative": "Western style, traditional cut, " + BASE_NEGATIVE,
        "category": "trendy",
        "gender": "male"
    }

    # ==================== ALL STYLES DICT ====================
    ALL_STYLES = {
        # Male styles
        "short_undercut": SHORT_UNDERCUT,
        "buzz_cut": BUZZ_CUT,
        "man_bun": MAN_BUN,
        "side_part": SIDE_PART,
        "slicked_back": SLICKED_BACK,
        "korean_style": KOREAN_STYLE,

        # Female styles
        "bob_cut": BOB_CUT,
        "pixie_cut": PIXIE_CUT,

        # Unisex styles
        "curly_afro": CURLY_AFRO,
        "blue_hair": BLUE_HAIR,
    }

    HAIR_STYLES = ALL_STYLES

    # ==================== METHODS ====================
    @classmethod
    def get_style(cls, style_id: str) -> Dict:
        """Get a specific style by ID"""
        if style_id not in cls.ALL_STYLES:
            raise ValueError(f"Style '{style_id}' not found")
        return cls.ALL_STYLES[style_id]

    @classmethod
    def get_all_styles(cls) -> Dict:
        """Get all styles dictionary"""
        return cls.ALL_STYLES

    @classmethod
    def get_style_list(cls) -> list:
        """Get list of all style IDs"""
        return list(cls.ALL_STYLES.keys())

    @classmethod
    def get_styles_by_category(cls, category: str) -> Dict:
        """Get styles filtered by category"""
        return {
            style_id: style_data
            for style_id, style_data in cls.ALL_STYLES.items()
            if style_data["category"] == category
        }

    @classmethod
    def get_styles_by_gender(cls, gender: str) -> Dict:
        """Get styles filtered by gender"""
        return {
            style_id: style_data
            for style_id, style_data in cls.ALL_STYLES.items()
            if style_data["gender"] == gender or style_data["gender"] == "unisex"
        }

    @classmethod
    def search_styles(cls, keyword: str) -> Dict:
        """Search styles by keyword in name or prompt"""
        keyword = keyword.lower()
        return {
            style_id: style_data
            for style_id, style_data in cls.ALL_STYLES.items()
            if keyword in style_data["name"].lower() or keyword in style_data["prompt"].lower()
        }

    @classmethod
    def get_style_info(cls, style_id: str) -> Dict[str, str]:
        """Get formatted style information for API response"""
        if style_id not in cls.ALL_STYLES:
            raise ValueError(f"Style '{style_id}' not found")

        style = cls.ALL_STYLES[style_id]
        return {
            "id": style["id"],
            "name": style["name"],
            "prompt": style["prompt"],
            "negative": style["negative"],
            "category": style["category"],
            "gender": style["gender"]
        }

    @classmethod
    def get_prompt_only(cls, style_id: str) -> tuple:
        """Get only prompt and negative prompt for generation"""
        style = cls.get_style(style_id)
        return (style["prompt"], style["negative"])


# ==================== QUICK ACCESS ====================
HAIR_STYLES = HairStylePrompts.ALL_STYLES