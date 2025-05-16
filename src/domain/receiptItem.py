from dataclasses import dataclass
from typing import List, Optional

@dataclass
class MenuItem:
    nm: Optional[str] = None
    num: Optional[str] = None
    unitprice: Optional[str] = None
    cnt: Optional[str] = None
    discountprice: Optional[str] = None
    price: Optional[str] = None
    itemsubtotal: Optional[str] = None
    vatyn: Optional[str] = None
    etc: Optional[str] = None
    sub_nm: Optional[str] = None
    sub_num: Optional[str] = None
    sub_unitprice: Optional[str] = None
    sub_cnt: Optional[str] = None
    sub_discountprice: Optional[str] = None
    sub_price: Optional[str] = None
    sub_etc: Optional[str] = None

@dataclass
class VoidMenuItem:
    nm: Optional[str] = None
    num: Optional[str] = None
    unitprice: Optional[str] = None
    cnt: Optional[str] = None
    price: Optional[str] = None
    etc: Optional[str] = None

@dataclass
class Subtotal:
    subtotal_price: Optional[str] = None
    discount_price: Optional[str] = None
    subtotal_count: Optional[str] = None
    service_price: Optional[str] = None
    othersvc_price: Optional[str] = None
    tax_price: Optional[str] = None
    tax_and_service: Optional[str] = None
    etc: Optional[str] = None

@dataclass
class VoidTotal:
    subtotal_price: Optional[str] = None
    tax_price: Optional[str] = None
    total_price: Optional[str] = None
    etc: Optional[str] = None

@dataclass
class Total:
    total_price: Optional[str] = None
    total_etc: Optional[str] = None
    cashprice: Optional[str] = None
    changeprice: Optional[str] = None
    creditcardprice: Optional[str] = None
    emoneyprice: Optional[str] = None
    menutype_cnt: Optional[str] = None
    menuqty_cnt: Optional[str] = None

@dataclass
class ReceiptItem:
    menu_items: List[MenuItem]
    void_menu_items: List[VoidMenuItem]
    subtotal: Optional[Subtotal] = None
    void_total: Optional[VoidTotal] = None
    total: Optional[Total] = None
    IMG_PATH: Optional[str] = None # 画像ファイルのファイル名かパス

    def xml(self) -> str:
        def tag(name, value):
            return f"<{name}>{value}</{name}>" if value else ""
        
        parts = ["<s>"]
        for m in self.menu_items:
            parts.extend([
                tag("s_menu_nm", m.nm),
                tag("s_menu_num", m.num),
                tag("s_menu_unitprice", m.unitprice),
                tag("s_menu_cnt", m.cnt),
                tag("s_menu_discountprice", m.discountprice),
                tag("s_menu_price", m.price),
                tag("s_menu_itemsubtotal", m.itemsubtotal),
                tag("s_menu_vatyn", m.vatyn),
                tag("s_menu_etc", m.etc),
                tag("s_menu_sub_nm", m.sub_nm),
                tag("s_menu_sub_num", m.sub_num),
                tag("s_menu_sub_unitprice", m.sub_unitprice),
                tag("s_menu_sub_cnt", m.sub_cnt),
                tag("s_menu_sub_discountprice", m.sub_discountprice),
                tag("s_menu_sub_price", m.sub_price),
                tag("s_menu_sub_etc", m.sub_etc)
            ])
        for v in self.void_menu_items:
            parts.extend([
                tag("s_void_menu_nm", v.nm),
                tag("s_void_menu_num", v.num),
                tag("s_void_menu_unitprice", v.unitprice),
                tag("s_void_menu_cnt", v.cnt),
                tag("s_void_menu_price", v.price),
                tag("s_void_menu_etc", v.etc)
            ])
        if self.subtotal:
            parts.extend([
                tag("s_subtotal_price", self.subtotal.subtotal_price),
                tag("s_subtotal_discount_price", self.subtotal.discount_price),
                tag("s_subtotal_count", self.subtotal.subtotal_count),
                tag("s_service_price", self.subtotal.service_price),
                tag("s_othersvc_price", self.subtotal.othersvc_price),
                tag("s_tax_price", self.subtotal.tax_price),
                tag("s_tax_and_service", self.subtotal.tax_and_service),
                tag("s_subtotal_etc", self.subtotal.etc)
            ])
        if self.void_total:
            parts.extend([
                tag("s_voidtotal_subtotal_price", self.void_total.subtotal_price),
                tag("s_voidtotal_tax_price", self.void_total.tax_price),
                tag("s_voidtotal_total_price", self.void_total.total_price),
                tag("s_voidtotal_etc", self.void_total.etc)
            ])
        if self.total:
            parts.extend([
                tag("s_total_total_price", self.total.total_price),
                tag("s_total_total_etc", self.total.total_etc),
                tag("s_total_cashprice", self.total.cashprice),
                tag("s_total_changeprice", self.total.changeprice),
                tag("s_total_creditcardprice", self.total.creditcardprice),
                tag("s_total_emoneyprice", self.total.emoneyprice),
                tag("s_total_menutype_cnt", self.total.menutype_cnt),
                tag("s_total_menuqty_cnt", self.total.menuqty_cnt)
            ])
        parts.append("</s>")
        return "\n".join(parts)
    
    @classmethod
    def get_xml_tags(cls) -> list[str]:
        return [
            "<s>",
            
            # menu
            "<s_menu_nm>", "</s_menu_nm>",
            "<s_menu_num>", "</s_menu_num>",
            "<s_menu_unitprice>", "</s_menu_unitprice>",
            "<s_menu_cnt>", "</s_menu_cnt>",
            "<s_menu_discountprice>", "</s_menu_discountprice>",
            "<s_menu_price>", "</s_menu_price>",
            "<s_menu_itemsubtotal>", "</s_menu_itemsubtotal>",
            "<s_menu_vatyn>", "</s_menu_vatyn>",
            "<s_menu_etc>", "</s_menu_etc>",
            "<s_menu_sub_nm>", "</s_menu_sub_nm>",
            "<s_menu_sub_num>", "</s_menu_sub_num>",
            "<s_menu_sub_unitprice>", "</s_menu_sub_unitprice>",
            "<s_menu_sub_cnt>", "</s_menu_sub_cnt>",
            "<s_menu_sub_discountprice>", "</s_menu_sub_discountprice>",
            "<s_menu_sub_price>", "</s_menu_sub_price>",
            "<s_menu_sub_etc>", "</s_menu_sub_etc>",

            # void_menu
            "<s_void_menu_nm>", "</s_void_menu_nm>",
            "<s_void_menu_num>", "</s_void_menu_num>",
            "<s_void_menu_unitprice>", "</s_void_menu_unitprice>",
            "<s_void_menu_cnt>", "</s_void_menu_cnt>",
            "<s_void_menu_price>", "</s_void_menu_price>",
            "<s_void_menu_etc>", "</s_void_menu_etc>",

            # subtotal
            "<s_subtotal_price>", "</s_subtotal_price>",
            "<s_subtotal_discount_price>", "</s_subtotal_discount_price>",
            "<s_subtotal_count>", "</s_subtotal_count>",
            "<s_service_price>", "</s_service_price>",
            "<s_othersvc_price>", "</s_othersvc_price>",
            "<s_tax_price>", "</s_tax_price>",
            "<s_tax_and_service>", "</s_tax_and_service>",
            "<s_subtotal_etc>", "</s_subtotal_etc>",

            # void total
            "<s_voidtotal_subtotal_price>", "</s_voidtotal_subtotal_price>",
            "<s_voidtotal_tax_price>", "</s_voidtotal_tax_price>",
            "<s_voidtotal_total_price>", "</s_voidtotal_total_price>",
            "<s_voidtotal_etc>", "</s_voidtotal_etc>",

            # total
            "<s_total_total_price>", "</s_total_total_price>",
            "<s_total_total_etc>", "</s_total_total_etc>",
            "<s_total_cashprice>", "</s_total_cashprice>",
            "<s_total_changeprice>", "</s_total_changeprice>",
            "<s_total_creditcardprice>", "</s_total_creditcardprice>",
            "<s_total_emoneyprice>", "</s_total_emoneyprice>",
            "<s_total_menutype_cnt>", "</s_total_menutype_cnt>",
            "<s_total_menuqty_cnt>", "</s_total_menuqty_cnt>",

            "</s>"
        ]

