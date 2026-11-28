package org.headcontrol.tracker;

public final class ConsensusPointerTest {
    private static final StablePointer.Profile OLD = StablePointer.Profile.PRECISE;
    private static final StablePointer.Profile NEW = StablePointer.Profile.CONSENSUS;
    private static final StablePointer.Pose NEUTRAL = new StablePointer.Pose(0,0,0,4,3);

    private static void require(boolean ok, String why) {
        if (!ok) throw new AssertionError(why);
    }

    private static void step(StablePointer p, StablePointer.Profile profile, long time,
                             double position, double angle, boolean edge) {
        p.update(time, new StablePointer.Pose(angle,0,0,4,3),
                position, 500, p.x, p.y, 1, profile, false, edge, false);
    }

    private static StablePointer anchor(StablePointer.Profile profile, double position) {
        var p = new StablePointer();
        for (int i = 0; i <= 8; i++)
            p.update(40L*i, NEUTRAL, position, 500, p.x, p.y, 1, profile, false, false, false);
        require(p.state == StablePointer.State.ANCHORED, "setup anchor");
        return p;
    }

    private static long small(StablePointer.Profile profile, double dp) {
        var p = anchor(profile,400);
        long first = -1;
        for (int i=0; i<12; i++) {
            step(p,profile,400+40L*i,400+dp,.5,false);
            if (first < 0 && p.x>400.5) first = 40L*i;
        }
        require(first < 0 || p.x == 400+dp, "fine step reaches its exact target");
        return first;
    }

    public static void main(String[] args) {
        var old = anchor(OLD,400);
        var candidate = anchor(NEW,400);
        for (int i=0; i<4; i++) {
            step(old,OLD,400+40L*i,415,.55,false);
            step(candidate,NEW,400+40L*i,415,.55,false);
        }
        require(old.x>400 && candidate.x==400, "weak large transient is blocked");
        step(candidate,NEW,560,400,0,false);
        step(candidate,NEW,600,440,2,false);
        step(candidate,NEW,640,400,0,false);
        require(candidate.x==400 && candidate.state==StablePointer.State.ANCHORED, "isolated spike never releases");

        var coherent = anchor(NEW,400);
        for (int i=0; i<6; i++) step(coherent,NEW,400+40L*i,415,.45+i*.045,false);
        require(coherent.x>400, "weak large movement with directional progress can release");
        var opposed = anchor(NEW,400);
        for (int i=0; i<9; i++) step(opposed,NEW,400+40L*i,415,i%2==0 ? 1 : -1,false);
        require(opposed.x==400, "opposing facial directions do not repeatedly unlock");

        require(small(OLD,3) < 0 && small(NEW,3)==80, "3 dp precision preserved in candidate");
        require(small(NEW,6)==small(OLD,6) && small(NEW,6)==80, "6 dp no added confirmation");
        var edge = anchor(NEW,2);
        for (int i=0; i<10; i++) step(edge,NEW,400+40L*i,0,.5,true);
        require(edge.x==0, "edge still exact");
        System.out.println("ConsensusPointerTest OK; synthetic 3 and 6 dp first movement: 80 ms");
    }
}
