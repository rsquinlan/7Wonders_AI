#ifndef CARD_H
#define CARD_H

#include <string>
#include <vector>

namespace CARD_ID{
        enum ID{
			// Materials
            lumber_yard,
            stone_pit,
            clay_pool,
            ore_vein,
            tree_farm,
            excavation,
            clay_pit,
            timber_yard,
            forest_cave,
            mine,
            sawmill,
            quarry,
            brickyard,
            foundry,

			//Manufactured
            loom,
            glassworks,
            press,

			//Civilian
            altar,
            theater,
            pawnshop,
            baths,
            temple,
            courthouse,
            statue,
            aqueduct,
            gardens,
            town_hall,
            senate,
            pantheon,
            palace,

			//Comercial
            tavern,
            east_trading_post,
            west_trading_post,
            marketplace,
            forum,
            caravansery,
            vineyard,
            bazar,
            haven,
            lighthouse,
            chamber_of_commerce,
            arena,

			//Military
            stockade,
            barracks,
            guard_tower,
            walls,
            training_ground,
            stables,
            archery_range,
            fortifications,
            circus,
            arsenal,
            siege_workshop,

			//Scientific
            apothecary,
            workshop,
            scriptorium,
            dispensary,
            laboratory,
            library,
            school,
            lodge,
            observatory,
            university,
            academy,
            study,

			//Guild
            workers,
            craftsmens,
            traders,
            philosophers,
            spies,
            magistrates,
            shipowners,
            strategists,
            scientists,
            builders,
        };

}


namespace CARD_TYPE {
enum type{
    materials,
    manufactured,
    civilian,
    commercial,
    military,
    scientific,
    guild,
    };
}

namespace DMAG {

class Card
{
private:
	int id;
    std::string name;
	int type;
    int era;
    int amount[5];
	int resource_type;
	std::vector<int> resource_list;

public:
    Card();
    Card(int id, std::string name, int type, int era, std::vector<int> amount);
    [[nodiscard]] std::string GetName() const;
    [[nodiscard]] int GetId() const;
    int GetType() const;
    int GetEra() const;
    int GetAmount(int) const;
    bool Equal(Card c);
};

}
#endif // CARD_H
