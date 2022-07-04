
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <set>
#include "assert.h"

#ifndef SCOREBOARD_H_
#define SCOREBOARD_H_


#include "opucore.h"
namespace opu {
class warp_inst_t;

class Scoreboard {
public:
  Scoreboard(unsigned sid, unsigned n_warps/*, class gpgpu_t *gpu*/);

    void reserveRegisters(const warp_inst_t *inst);
    void releaseRegisters(const warp_inst_t *inst);
    void releaseRegister(unsigned wid, unsigned regnum);

    bool checkCollision(unsigned wid, const warp_inst_t *inst) const;
    bool pendingWrites(unsigned wid) const;
    void printContents() const;
    // TODO schi compile warn on const const bool islongop(unsigned warp_id, unsigned regnum);
    bool islongop(unsigned warp_id, unsigned regnum);
private:
    void reserveRegister(unsigned wid, unsigned regnum);
    int get_sid() const { return m_sid; }

    unsigned m_sid;

    // keeps track of pending writes to registers
    // indexed by warp id, reg_id => pending write count
    std::vector< std::set<unsigned> > reg_table;
    //Register that depend on a long operation (global, local or tex memory)
    std::vector< std::set<unsigned> > longopregs;
    // class gpgpu_t *m_gpu;
};

}

#endif /* SCOREBOARD_H_ */
